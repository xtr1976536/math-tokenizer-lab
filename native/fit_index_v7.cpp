/* Exact raw adjacent/non-overlap counts and pair replacement for fit-only RL.
 * No lexical rules, reward, tree or corpus state here. Python owns index heap. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <unordered_map>
#include <vector>
#include <cstdint>

using Counts=std::unordered_map<uint64_t,long>;
static PyObject* dictionary(const Counts&c){
    PyObject*d=PyDict_New();if(!d)return NULL;
    for(auto &entry:c){
        PyObject*k=Py_BuildValue("LL",(long long)(entry.first>>32),(long long)(entry.first&0xffffffffu));
        PyObject*v=PyLong_FromLong(entry.second);
        if(!k||!v||PyDict_SetItem(d,k,v)<0){Py_XDECREF(k);Py_XDECREF(v);Py_DECREF(d);return NULL;}
        Py_DECREF(k);Py_DECREF(v);
    }return d;
}
static void count(const std::vector<long>&ids,Counts&raw,Counts&gains){
    std::unordered_map<uint64_t,size_t>last;
    for(size_t i=0;i+1<ids.size();i++){
        uint64_t k=((uint64_t)ids[i]<<32)|(uint64_t)ids[i+1];raw[k]++;
        if(ids[i]==ids[i+1]){
            auto previous=last.find(k);if(previous!=last.end()&&previous->second+1==i)continue;last[k]=i;
        }gains[k]++;
    }
}
static bool vector(PyObject*obj,std::vector<long>&out){
    if(!PyList_Check(obj)){PyErr_SetString(PyExc_TypeError,"index input must be ID lists");return false;}
    Py_ssize_t n=PyList_Size(obj);out.reserve(n);
    for(Py_ssize_t i=0;i<n;i++){long x=PyLong_AsLong(PyList_GetItem(obj,i));if(PyErr_Occurred())return false;out.push_back(x);}return true;
}
static PyObject* list(const std::vector<long>&a){
    PyObject*out=PyList_New(a.size());if(!out)return NULL;
    for(size_t i=0;i<a.size();i++){PyObject*v=PyLong_FromLong(a[i]);if(!v){Py_DECREF(out);return NULL;}PyList_SET_ITEM(out,i,v);}return out;
}
static PyObject* scan(PyObject*self,PyObject*arg){
    std::vector<long>ids;if(!vector(arg,ids))return NULL;Counts raw,gain;count(ids,raw,gain);
    PyObject*a=dictionary(raw),*b=dictionary(gain);if(!a||!b){Py_XDECREF(a);Py_XDECREF(b);return NULL;}return Py_BuildValue("NN",a,b);
}
static PyObject* merge(PyObject*self,PyObject*args){
    PyObject*ia,*sa,*ea;long a,b,newid;
    if(!PyArg_ParseTuple(args,"OOOlll",&ia,&sa,&ea,&a,&b,&newid))return NULL;
    std::vector<long>ids,starts,ends;if(!vector(ia,ids)||!vector(sa,starts)||!vector(ea,ends))return NULL;
    if(ids.size()!=starts.size()||ids.size()!=ends.size()){PyErr_SetString(PyExc_ValueError,"ID/source span lengths differ");return NULL;}
    std::vector<long>out,s,e;out.reserve(ids.size());s.reserve(ids.size());e.reserve(ids.size());
    for(size_t i=0;i<ids.size();){
        s.push_back(starts[i]);
        if(i+1<ids.size()&&ids[i]==a&&ids[i+1]==b){out.push_back(newid);e.push_back(ends[i+1]);i+=2;}
        else{out.push_back(ids[i]);e.push_back(ends[i]);i++;}
    }
    Counts raw,gain;count(out,raw,gain);
    PyObject*x=list(out),*y=list(s),*z=list(e),*r=dictionary(raw),*g=dictionary(gain);
    if(!x||!y||!z||!r||!g){Py_XDECREF(x);Py_XDECREF(y);Py_XDECREF(z);Py_XDECREF(r);Py_XDECREF(g);return NULL;}
    return Py_BuildValue("NNNNN",x,y,z,r,g);
}
static PyMethodDef methods[]={{"scan",scan,METH_O,"Exact raw/nonoverlap pair count."},{"merge",merge,METH_VARARGS,"Exact pair merge and recount."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_fit_index_v7",NULL,-1,methods};
PyMODINIT_FUNC PyInit__fit_index_v7(void){return PyModule_Create(&module);}
