/* Joint path code: normalized token-transition costs + optional byte signal.
 * Uses exact finite-state DP, with sparse reachable predecessor states.
 * No syntactic parser, command table, or evaluation labels. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <math.h>

static PyObject *encode(PyObject*self,PyObject*args){
 PyObject *text,*children,*terminal,*lengths,*costs,*signal;int maxlen,V,with_cost=0;double rate;
 if(!PyArg_ParseTuple(args,"OOOOOOiid|p",&text,&children,&terminal,&lengths,&costs,&signal,&maxlen,&V,&rate,&with_cost))return NULL;
 if(!PyUnicode_Check(text)||V<256||maxlen<1||!isfinite(rate)||rate<0){PyErr_SetString(PyExc_ValueError,"invalid path configuration");return NULL;}
 Py_ssize_t len;const uint8_t*raw=(const uint8_t*)PyUnicode_AsUTF8AndSize(text,&len);if(!raw)return NULL;
 if(len>1000000){PyErr_SetString(PyExc_ValueError,"million-byte limit");return NULL;}
 int n=(int)len;Py_buffer cb={0},tb={0},lb={0},wb={0},sb={0};PyObject*result=NULL;
 if(PyObject_GetBuffer(children,&cb,PyBUF_SIMPLE)<0)return NULL;
 if(PyObject_GetBuffer(terminal,&tb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(lengths,&lb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(costs,&wb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(signal,&sb,PyBUF_SIMPLE)<0)goto buffers;
 if(cb.len%1024||tb.len%4||cb.len/1024!=tb.len/4||lb.len!=V*4||wb.len!=(Py_ssize_t)(V+2)*(V+1)*8||sb.len!=257*257*8){PyErr_SetString(PyExc_ValueError,"invalid buffers");goto buffers;}
 const int32_t*ch=cb.buf,*term=tb.buf,*lens=lb.buf;const double*w=wb.buf,*s=sb.buf;int nodes=(int)(tb.len/4);
 size_t cap=(size_t)n*maxlen+1;if(cap>20000000){PyErr_SetString(PyExc_ValueError,"candidate memory bound exceeded");goto buffers;}
 int *a=PyMem_Malloc(cap*4),*b=PyMem_Malloc(cap*4),*ids=PyMem_Malloc(cap*4),*pred=PyMem_Malloc(cap*4),*next=PyMem_Malloc(cap*4);
 int *head=PyMem_Malloc(((size_t)n+1)*4);double*best=PyMem_Malloc(cap*8);
 if(!a||!b||!ids||!pred||!next||!head||!best){PyErr_NoMemory();goto memory;}
 for(int i=0;i<=n;i++)head[i]=-1;int count=0;
 for(int start=0;start<n;start++){
  int node=0;
  for(int end=start;end<n&&end-start<maxlen;end++){
   node=ch[(size_t)node*256+raw[end]];if(node<0)break;
   if(node>=nodes){PyErr_SetString(PyExc_ValueError,"invalid trie");goto memory;}
   int id=term[node];if(id<0)continue;
   if(id>=V||lens[id]!=end+1-start){PyErr_SetString(PyExc_ValueError,"invalid payload length");goto memory;}
   double val=INFINITY;int previous=-1;
   if(start==0)val=w[(size_t)(V+1)*(V+1)+id];
   else for(int k=head[start];k>=0;k=next[k]){
    double candidate=best[k]+w[(size_t)ids[k]*(V+1)+id];
    if(candidate<val-1e-12){val=candidate;previous=k;}
   }
   int e=end+1,c=(e>=2?raw[e-2]:256)*257+raw[e-1];
   a[count]=start;b[count]=e;ids[count]=id;pred[count]=previous;
   best[count]=val+rate+(e<n?s[c]:0);next[count]=head[e];head[e]=count;count++;
  }
 }
 double value=INFINITY;int final=-1;
 if(!n)value=w[(size_t)(V+1)*(V+1)+V];
 for(int k=head[n];k>=0;k=next[k]){
  double candidate=best[k]+w[(size_t)ids[k]*(V+1)+V];
  if(candidate<value-1e-12){value=candidate;final=k;}
 }
 if(n&&final<0){PyErr_SetString(PyExc_RuntimeError,"no lossless path");goto memory;}
 int tokens=0;for(int k=final;k>=0;k=pred[k])tokens++;
 result=PyList_New(tokens);if(!result)goto memory;
 for(int k=final,i=tokens-1;k>=0;k=pred[k],i--){PyObject*id=PyLong_FromLong(ids[k]);if(!id){Py_CLEAR(result);goto memory;}PyList_SET_ITEM(result,i,id);}
 if(with_cost)result=Py_BuildValue("Nd",result,value);
memory:
 PyMem_Free(a);PyMem_Free(b);PyMem_Free(ids);PyMem_Free(pred);PyMem_Free(next);PyMem_Free(head);PyMem_Free(best);
buffers:
 if(cb.obj)PyBuffer_Release(&cb);if(tb.obj)PyBuffer_Release(&tb);if(lb.obj)PyBuffer_Release(&lb);if(wb.obj)PyBuffer_Release(&wb);if(sb.obj)PyBuffer_Release(&sb);return result;
}
static PyMethodDef methods[]={{"encode",encode,METH_VARARGS,"Exact predictive path"},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_predictive_stage2",NULL,-1,methods};
PyMODINIT_FUNC PyInit__predictive_stage2(void){return PyModule_Create(&module);}
