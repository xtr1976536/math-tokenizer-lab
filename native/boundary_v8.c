/* Full raw-byte interface, lexical protection and additive brace-boundary
 * distortion. No evaluation reference parser or corpus data at runtime. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "frontend_v6.c"

static PyObject *encode(PyObject*self,PyObject*args){
    PyObject *text,*children,*terminal;int maxlen,with_cost=0;double tree,boundary;
    if(!PyArg_ParseTuple(args,"OOOidd|p",&text,&children,&terminal,&maxlen,&tree,&boundary,&with_cost))return NULL;
    if(!PyUnicode_Check(text)||!isfinite(tree)||tree<0||!isfinite(boundary)||boundary<0||maxlen<1){PyErr_SetString(PyExc_ValueError,"invalid text or distortion weight");return NULL;}
    Py_ssize_t length;const uint8_t *raw=(const uint8_t*)PyUnicode_AsUTF8AndSize(text,&length);
    if(!raw)return NULL;
    if(length>1000000){PyErr_SetString(PyExc_ValueError,"input exceeds documented million-byte limit");return NULL;}
    int n=(int)length;Py_buffer cb,tb;
    if(PyObject_GetBuffer(children,&cb,PyBUF_SIMPLE)<0)return NULL;
    if(PyObject_GetBuffer(terminal,&tb,PyBUF_SIMPLE)<0){PyBuffer_Release(&cb);return NULL;}
    if(cb.len%1024||tb.len%4||cb.len/1024!=tb.len/4){PyBuffer_Release(&cb);PyBuffer_Release(&tb);PyErr_SetString(PyExc_ValueError,"invalid trie buffers");return NULL;}
    int32_t *spans=PyMem_Malloc(((size_t)n*16+2)*4),*choice=PyMem_Malloc(((size_t)n+1)*4),*ids=PyMem_Malloc(((size_t)n+1)*4);
    int *maxend=PyMem_Calloc((size_t)n+1,sizeof(int)),*minstart=PyMem_Malloc(((size_t)n+1)*sizeof(int));
    int *stack=PyMem_Malloc(((size_t)n+1)*sizeof(int)),*prefix=PyMem_Calloc((size_t)n+2,sizeof(int));
    uint8_t *allowed=PyMem_Malloc((size_t)n+1),*positions=PyMem_Calloc((size_t)n+1,1);
    double *best=PyMem_Malloc(((size_t)n+1)*sizeof(double));PyObject *result=NULL;
    if(!spans||!choice||!ids||!maxend||!minstart||!allowed||!best||!stack||!prefix||!positions){PyErr_NoMemory();goto cleanup;}
    int ns=mt_tree(raw,n,spans);if(ns<0){PyErr_NoMemory();goto cleanup;}
    const int32_t *ch=cb.buf,*term=tb.buf;mt_allowed(raw,n,ch,term,allowed);
    int depth=0;
    for(int a=0;a<n;){int b=unit_end(raw,n,a);
        if(raw[a]=='{'&&b==a+1)stack[depth++]=a;
        else if(raw[a]=='}'&&b==a+1&&depth){int start=stack[--depth];
            if(a>start+1){positions[start+1]=1;positions[a]=1;}}
        a=b;}
    for(int i=0;i<=n;i++){prefix[i+1]=prefix[i]+positions[i];minstart[i]=n+1;}
    for(int i=0;i<ns;i++){int a=spans[2*i],b=spans[2*i+1];if(b>maxend[a])maxend[a]=b;if(a<minstart[b])minstart[b]=a;}
    best[n]=0;
    for(int a=n-1;a>=0;a--){best[a]=INFINITY;choice[a]=-1;if(!allowed[a])continue;
        int node=0,mx=0,mn=n+1;
        for(int b=a;b<n&&b-a<maxlen;b++){
            node=ch[(size_t)node*256+raw[b]];if(node<0)break;
            int end=b+1,id=term[node];
            if(id>=0&&allowed[end]){
                int crossing=mx>end||mn<a;int missing=prefix[end]-prefix[a+1];
                double value=1.+tree*crossing+boundary*missing+best[end];
                if(value<best[a]-1e-12){best[a]=value;choice[a]=end;ids[a]=id;}
            }
            if(maxend[end]>mx)mx=maxend[end];if(minstart[end]<mn)mn=minstart[end];
        }
    }
    int pos=0,count=0;
    while(pos<n){if(choice[pos]<0){PyErr_SetString(PyExc_RuntimeError,"lexical lattice disconnected");goto cleanup;}count++;pos=choice[pos];}
    result=PyList_New(count);if(!result)goto cleanup;pos=0;
    for(int i=0;i<count;i++){PyObject*id=PyLong_FromLong(ids[pos]);if(!id){Py_CLEAR(result);goto cleanup;}PyList_SET_ITEM(result,i,id);pos=choice[pos];}
    if(with_cost){result=Py_BuildValue("Nd",result,best[0]);}
cleanup:
    PyMem_Free(spans);PyMem_Free(choice);PyMem_Free(ids);PyMem_Free(maxend);PyMem_Free(minstart);PyMem_Free(allowed);PyMem_Free(best);
    PyMem_Free(stack);PyMem_Free(prefix);PyMem_Free(positions);PyBuffer_Release(&cb);PyBuffer_Release(&tb);return result;
}
static PyMethodDef methods[]={{"encode",encode,METH_VARARGS,"Lossless additive boundary/tree shortest path"},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_boundary_v8",NULL,-1,methods};
PyMODINIT_FUNC PyInit__boundary_v8(void){return PyModule_Create(&module);}
