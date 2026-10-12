/* Source-tree rate/distortion lattice: lossless lexical masks, soft structure.
 * Same trie traversal as shortest decoder; only legal terminal edge weights
 * change. No reference KaTeX data, corpus cache or model selection at runtime. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "frontend_v6.c"

static PyObject *encode(PyObject*self,PyObject*args){
    PyObject *text,*children,*terminal,*risk=Py_None;int maxlen,with_cost=0,lexicographic=0;double penalty;
    if(!PyArg_ParseTuple(args,"OOOid|pOp",&text,&children,&terminal,&maxlen,&penalty,&with_cost,&risk,&lexicographic))return NULL;
    if(!PyUnicode_Check(text)||!isfinite(penalty)||penalty<0||maxlen<1){PyErr_SetString(PyExc_ValueError,"invalid text or structure penalty");return NULL;}
    Py_ssize_t length;const uint8_t *raw=(const uint8_t*)PyUnicode_AsUTF8AndSize(text,&length);
    if(!raw)return NULL;
    if(length>1000000){PyErr_SetString(PyExc_ValueError,"input exceeds documented million-byte limit");return NULL;}
    int n=(int)length;Py_buffer cb,tb;
    if(PyObject_GetBuffer(children,&cb,PyBUF_SIMPLE)<0)return NULL;
    if(PyObject_GetBuffer(terminal,&tb,PyBUF_SIMPLE)<0){PyBuffer_Release(&cb);return NULL;}
    if(cb.len%1024||tb.len%4||cb.len/1024!=tb.len/4){PyBuffer_Release(&cb);PyBuffer_Release(&tb);PyErr_SetString(PyExc_ValueError,"invalid trie buffers");return NULL;}
    int32_t *spans=PyMem_Malloc(((size_t)n*16+2)*4),*choice=PyMem_Malloc(((size_t)n+1)*4),*ids=PyMem_Malloc(((size_t)n+1)*4);
    int *maxend=PyMem_Calloc((size_t)n+1,sizeof(int)),*minstart=PyMem_Malloc(((size_t)n+1)*sizeof(int));
    uint8_t *allowed=PyMem_Malloc((size_t)n+1);double *best=PyMem_Malloc(((size_t)n+1)*sizeof(double));
    int *lengths=PyMem_Malloc(((size_t)n+1)*sizeof(int));
    PyObject *result=NULL;
    if(!spans||!choice||!ids||!maxend||!minstart||!allowed||!best||!lengths){PyErr_NoMemory();goto cleanup;}
    int ns=mt_tree(raw,n,spans);if(ns<0){PyErr_NoMemory();goto cleanup;}
    double risk_weights[8];
    if(risk!=Py_None){
        if(!PyList_Check(risk)||PyList_Size(risk)!=8){PyErr_SetString(PyExc_ValueError,"risk weights need 8 calibrated node categories");goto cleanup;}
        for(int j=0;j<8;j++){risk_weights[j]=PyFloat_AsDouble(PyList_GetItem(risk,j));if(PyErr_Occurred()||risk_weights[j]<0||!isfinite(risk_weights[j]))goto cleanup;}
    }
    const int32_t *ch=cb.buf,*term=tb.buf;
    mt_allowed(raw,n,ch,term,allowed);
    for(int i=0;i<=n;i++)minstart[i]=n+1;
    for(int i=0;i<ns;i++){int a=spans[2*i],b=spans[2*i+1];
        int bucket=(raw[a]=='{'?0:raw[a]=='\\'?1:2)*2+(b-a>8);
        if(risk!=Py_None && risk_weights[bucket]<=0)continue;
        if(b>maxend[a])maxend[a]=b;if(a<minstart[b])minstart[b]=a;}
    best[n]=0;lengths[n]=0;
    for(int a=n-1;a>=0;a--){
        best[a]=INFINITY;lengths[a]=n+1;choice[a]=-1;if(!allowed[a])continue;
        int node=0,mx=0,mn=n+1;
        for(int b=a;b<n&&b-a<maxlen;b++){
            node=ch[(size_t)node*256+raw[b]];if(node<0)break;
            int end=b+1,id=term[node];int crossing=mx>end||mn<a;
            if(id>=0&&allowed[end]){
                double distortion=crossing;
                if(crossing&&risk!=Py_None){
                    distortion=0.;
                    for(int k=0;k<ns;k++){
                        int s=spans[2*k],e=spans[2*k+1];
                        if((a<s&&s<end&&end<e)||(s<a&&a<e&&e<end)){
                            int bucket=(raw[s]=='{'?0:raw[s]=='\\'?1:2)*2+(e-s>8);
                            if(risk_weights[bucket]>distortion)distortion=risk_weights[bucket];
                        }
                    }
                }
                double value=(lexicographic?distortion:1.+penalty*distortion)+best[end];
                int token_length=1+lengths[end];
                int better=lexicographic?(token_length<lengths[a]||(token_length==lengths[a]&&value<best[a]-1e-12)):(value<best[a]-1e-12);
                if(better){best[a]=value;lengths[a]=token_length;choice[a]=end;ids[a]=id;}
            }
            if(maxend[end]>mx)mx=maxend[end];if(minstart[end]<mn)mn=minstart[end];
        }
    }
    int pos=0,count=0;
    while(pos<n){if(choice[pos]<0){PyErr_SetString(PyExc_RuntimeError,"lexical lattice disconnected");goto cleanup;}count++;pos=choice[pos];}
    result=PyList_New(count);if(!result)goto cleanup;pos=0;
    for(int i=0;i<count;i++){PyObject*id=PyLong_FromLong(ids[pos]);if(!id){Py_CLEAR(result);goto cleanup;}PyList_SET_ITEM(result,i,id);pos=choice[pos];}
    if(with_cost){PyObject*tuple=Py_BuildValue("Nd",result,best[0]);result=tuple;}
cleanup:
    PyMem_Free(spans);PyMem_Free(choice);PyMem_Free(ids);PyMem_Free(maxend);PyMem_Free(minstart);PyMem_Free(allowed);PyMem_Free(best);PyMem_Free(lengths);
    PyBuffer_Release(&cb);PyBuffer_Release(&tb);return result;
}
static PyMethodDef methods[]={{"encode",encode,METH_VARARGS,"Lossless source-tree weighted shortest segmentation."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_balanced_v7",NULL,-1,methods};
PyMODINIT_FUNC PyInit__balanced_v7(void){return PyModule_Create(&module);}
