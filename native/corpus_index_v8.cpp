/* Exact corpus-level adjacency heap and nonoverlap gains for fit-only search.
 * Algorithm equals Indexed; avoiding per-formula Python dictionaries is an
 * engineering optimization, never credited as an RL learning gain. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <new>
using Key=uint64_t;using Counts=std::unordered_map<Key,long>;
struct Local{std::vector<long> ids;Counts raw,gain;};
struct Item{long count;Key pair;unsigned version;
    bool operator<(const Item&o)const{if(count!=o.count)return count<o.count;return pair>o.pair;}};
struct Index{std::vector<Local> seqs;Counts raw,gain;std::unordered_map<Key,std::unordered_set<size_t>> occurrences;
    std::unordered_map<Key,unsigned> version;std::priority_queue<Item> heap;};
static Key key(long a,long b){return ((Key)a<<32)|(Key)b;}
static void count(Local&s){
    s.raw.clear();s.gain.clear();std::unordered_map<Key,size_t>last;
    for(size_t i=0;i+1<s.ids.size();i++){Key p=key(s.ids[i],s.ids[i+1]);s.raw[p]++;
        if(s.ids[i]==s.ids[i+1]){auto old=last.find(p);if(old!=last.end()&&old->second+1==i)continue;last[p]=i;}s.gain[p]++;}
}
static Index*get(PyObject*c){return (Index*)PyCapsule_GetPointer(c,"math_fit_index_v8");}
static void destroy(PyObject*c){delete get(c);}
static bool ids(PyObject*object,std::vector<long>&out){
    if(!PyList_Check(object)){PyErr_SetString(PyExc_TypeError,"ID sequences must be lists");return false;}
    for(Py_ssize_t i=0;i<PyList_Size(object);i++){long value=PyLong_AsLong(PyList_GetItem(object,i));
        if(PyErr_Occurred())return false;
        if(value<0||value>INT_MAX){PyErr_SetString(PyExc_ValueError,"ID outside supported nonnegative range");return false;}out.push_back(value);}
    return true;
}
static PyObject*init(PyObject*,PyObject*object){
    if(!PyList_Check(object)){PyErr_SetString(PyExc_TypeError,"corpus must be a list");return NULL;}
    Index*x=NULL;
    try{x=new Index;
        for(Py_ssize_t i=0;i<PyList_Size(object);i++){
            Local s;if(!ids(PyList_GetItem(object,i),s.ids)){delete x;return NULL;}count(s);
            for(auto&e:s.raw){x->raw[e.first]+=e.second;x->occurrences[e.first].insert(i);}
            for(auto&e:s.gain)x->gain[e.first]+=e.second;x->seqs.push_back(std::move(s));}
        for(auto&e:x->raw)x->heap.push({e.second,e.first,0});
        PyObject*c=PyCapsule_New(x,"math_fit_index_v8",destroy);if(!c)delete x;return c;
    }catch(const std::bad_alloc&){delete x;return PyErr_NoMemory();}
}
static PyObject*top(PyObject*,PyObject*args){
    PyObject*c;int k;long minimum;
    if(!PyArg_ParseTuple(args,"Oil",&c,&k,&minimum))return NULL;Index*x=get(c);if(!x)return NULL;
    if(k<0||minimum<1){PyErr_SetString(PyExc_ValueError,"invalid candidate count/minimum");return NULL;}
    PyObject*out=PyList_New(0);if(!out)return NULL;
    try{std::vector<Item>held;
        while(!x->heap.empty()&&PyList_Size(out)<k){Item i=x->heap.top();x->heap.pop();
            if(x->version[i.pair]!=i.version||x->raw[i.pair]!=i.count)continue;
            held.push_back(i);if(i.count<minimum)break;
            PyObject*value=Py_BuildValue("lll",(long)(i.pair>>32),(long)(i.pair&0xffffffffu),x->gain[i.pair]);
            if(!value||PyList_Append(out,value)<0){Py_XDECREF(value);Py_DECREF(out);return NULL;}Py_DECREF(value);}
        for(auto&i:held)x->heap.push(i);return out;
    }catch(const std::bad_alloc&){Py_DECREF(out);return PyErr_NoMemory();}
}
static PyObject*merge(PyObject*,PyObject*args){
    PyObject*c;long a,b,newid;if(!PyArg_ParseTuple(args,"Olll",&c,&a,&b,&newid))return NULL;Index*x=get(c);if(!x)return NULL;
    if(a<0||b<0||newid<0||a>INT_MAX||b>INT_MAX||newid>INT_MAX){PyErr_SetString(PyExc_ValueError,"invalid merge IDs");return NULL;}
    try{Key target=key(a,b);std::vector<size_t>affected(x->occurrences[target].begin(),x->occurrences[target].end());
        std::sort(affected.begin(),affected.end());Counts changed;
        for(size_t i:affected){Local&s=x->seqs[i];Counts previous=std::move(s.raw),previousgain=std::move(s.gain);std::vector<long>out;
            out.reserve(s.ids.size());for(size_t j=0;j<s.ids.size();){
                if(j+1<s.ids.size()&&s.ids[j]==a&&s.ids[j+1]==b){out.push_back(newid);j+=2;}else out.push_back(s.ids[j++]);}
            s.ids=std::move(out);count(s);std::unordered_set<Key>all;
            for(auto&e:previous)all.insert(e.first);for(auto&e:s.raw)all.insert(e.first);
            for(Key p:all){if(changed.find(p)==changed.end())changed[p]=x->raw[p];
                long old=previous[p],current=s.raw.count(p)?s.raw[p]:0;
                x->raw[p]+=current-old;x->gain[p]+=(s.gain.count(p)?s.gain[p]:0)-previousgain[p];
                if(!current)x->occurrences[p].erase(i);else if(!old)x->occurrences[p].insert(i);}
        }
        for(auto&e:changed)if(x->raw[e.first]!=e.second){unsigned v=++x->version[e.first];if(x->raw[e.first]>0)x->heap.push({x->raw[e.first],e.first,v});}
        if(x->heap.size()>4*std::max((size_t)1,x->raw.size())){x->heap={};for(auto&e:x->raw)if(e.second>0)x->heap.push({e.second,e.first,x->version[e.first]});}
        Py_RETURN_NONE;
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
}
static PyObject*dict(const Counts&values){
    PyObject*out=PyDict_New();if(!out)return NULL;
    for(auto&e:values)if(e.second){PyObject*k=Py_BuildValue("ll",(long)(e.first>>32),(long)(e.first&0xffffffffu));PyObject*v=PyLong_FromLong(e.second);
        if(!k||!v||PyDict_SetItem(out,k,v)<0){Py_XDECREF(k);Py_XDECREF(v);Py_DECREF(out);return NULL;}Py_DECREF(k);Py_DECREF(v);}return out;
}
static PyObject*snapshot(PyObject*,PyObject*c){
    Index*x=get(c);if(!x)return NULL;PyObject*seqs=PyList_New(x->seqs.size());if(!seqs)return NULL;
    for(size_t i=0;i<x->seqs.size();i++){PyObject*row=PyList_New(x->seqs[i].ids.size());if(!row){Py_DECREF(seqs);return NULL;}
        for(size_t j=0;j<x->seqs[i].ids.size();j++){PyObject*v=PyLong_FromLong(x->seqs[i].ids[j]);if(!v){Py_DECREF(row);Py_DECREF(seqs);return NULL;}PyList_SET_ITEM(row,j,v);}PyList_SET_ITEM(seqs,i,row);}
    PyObject*r=dict(x->raw),*g=dict(x->gain);if(!r||!g){Py_DECREF(seqs);Py_XDECREF(r);Py_XDECREF(g);return NULL;}return Py_BuildValue("NNN",seqs,r,g);
}
static PyMethodDef methods[]={{"init",init,METH_O,"Initialize fit corpus."},{"top",top,METH_VARARGS,"Raw-frequency top pairs and exact nonoverlap gains."},
    {"merge",merge,METH_VARARGS,"Merge all affected formulas."},{"snapshot",snapshot,METH_O,"Audit sequence and count state."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_corpus_index_v8",NULL,-1,methods};
PyMODINIT_FUNC PyInit__corpus_index_v8(void){return PyModule_Create(&module);}
