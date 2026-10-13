/* Generic learned primitive protection and learned paired-byte scope signals.
 * No fixed TeX commands, delimiter characters, or mathematical AST parser. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <math.h>
static PyObject*encode(PyObject*self,PyObject*args){
 PyObject *text,*children,*terminal,*achildren,*aterminal,*pairs,*costs,*token_costs;int maxlen,with_cost=0;double scope;
 if(!PyArg_ParseTuple(args,"OOOOOOOOid|p",&text,&children,&terminal,&achildren,&aterminal,&pairs,&costs,&token_costs,&maxlen,&scope,&with_cost))return NULL;
 if(!PyUnicode_Check(text)||maxlen<1||scope<0||!isfinite(scope)){PyErr_SetString(PyExc_ValueError,"invalid learned scope configuration");return NULL;}
 Py_ssize_t length;const uint8_t*raw=(const uint8_t*)PyUnicode_AsUTF8AndSize(text,&length);if(!raw)return NULL;
 if(length>1000000){PyErr_SetString(PyExc_ValueError,"million-byte input limit");return NULL;}int n=(int)length;
 Py_buffer cb={0},tb={0},acb={0},atb={0},pb={0},wb={0},kb={0};PyObject*result=NULL;
 if(PyObject_GetBuffer(children,&cb,PyBUF_SIMPLE)<0)return NULL;
 if(PyObject_GetBuffer(terminal,&tb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(achildren,&acb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(aterminal,&atb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(pairs,&pb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(costs,&wb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(token_costs,&kb,PyBUF_SIMPLE)<0)goto buffers;
 if(cb.len%1024||tb.len%4||cb.len/1024!=tb.len/4||acb.len%1024||atb.len%4||acb.len/1024!=atb.len/4||pb.len%8||wb.len!=257*257*8||kb.len%8){PyErr_SetString(PyExc_ValueError,"invalid buffers");goto buffers;}
 const int32_t*ch=cb.buf,*term=tb.buf,*ach=acb.buf,*at=atb.buf,*pair=pb.buf;
 const double*w=wb.buf,*kcost=kb.buf;int nodes=tb.len/4,anodes=atb.len/4,V=kb.len/8;
 uint8_t*allowed=PyMem_Malloc((size_t)n+1),*marked=PyMem_Calloc((size_t)n+1,1);
 int *prefix=PyMem_Calloc((size_t)n+2,4),*stack=PyMem_Malloc(((size_t)n+1)*4),*end=PyMem_Malloc(((size_t)n+1)*4),*ids=PyMem_Malloc(((size_t)n+1)*4);
 double*best=PyMem_Malloc(((size_t)n+1)*8);
 if(!allowed||!marked||!prefix||!stack||!end||!ids||!best){PyErr_NoMemory();goto memory;}
 memset(allowed,1,(size_t)n+1);
 for(int a=0;a<n;){int node=0,last=a+1;
  for(int b=a;b<n&&b-a<maxlen;b++){node=ach[(size_t)node*256+raw[b]];if(node<0)break;if(node>=anodes){PyErr_SetString(PyExc_ValueError,"invalid primitive trie");goto memory;}if(at[node]>=0)last=b+1;}
  for(int b=a+1;b<last;b++)allowed[b]=0;a=last;
 }
 for(int p=0;p<pb.len/8;p++){int depth=0;
  for(int a=0;a<n;a++){if(raw[a]==pair[2*p])stack[depth++]=a;
   else if(raw[a]==pair[2*p+1]&&depth){int start=stack[--depth];if(a>start+1){marked[start+1]=1;marked[a]=1;}}}
 }
 for(int a=0;a<=n;a++)prefix[a+1]=prefix[a]+marked[a];best[n]=0;
 for(int a=n-1;a>=0;a--){best[a]=INFINITY;end[a]=-1;if(!allowed[a])continue;int node=0;
  for(int b=a;b<n&&b-a<maxlen;b++){node=ch[(size_t)node*256+raw[b]];if(node<0)break;if(node>=nodes){PyErr_SetString(PyExc_ValueError,"invalid trie");goto memory;}
   int id=term[node],e=b+1;if(id>=0&&allowed[e]){if(id>=V){PyErr_SetString(PyExc_ValueError,"invalid token");goto memory;}
    int c=(e>=2?raw[e-2]:256)*257+raw[e-1];double value=1.+kcost[id]+scope*(prefix[e]-prefix[a+1])+(e<n?w[c]:0.)+best[e];
    if(value<best[a]-1e-12){best[a]=value;end[a]=e;ids[a]=id;}}
  }
 }
 int pos=0,count=0;while(pos<n){if(end[pos]<=pos){PyErr_SetString(PyExc_RuntimeError,"protected lattice disconnected");goto memory;}count++;pos=end[pos];}
 result=PyList_New(count);if(!result)goto memory;pos=0;
 for(int i=0;i<count;i++){PyObject*id=PyLong_FromLong(ids[pos]);if(!id){Py_CLEAR(result);goto memory;}PyList_SET_ITEM(result,i,id);pos=end[pos];}
 if(with_cost)result=Py_BuildValue("Nd",result,best[0]);
memory:
 PyMem_Free(allowed);PyMem_Free(marked);PyMem_Free(prefix);PyMem_Free(stack);PyMem_Free(end);PyMem_Free(ids);PyMem_Free(best);
buffers:
 if(cb.obj)PyBuffer_Release(&cb);if(tb.obj)PyBuffer_Release(&tb);if(acb.obj)PyBuffer_Release(&acb);if(atb.obj)PyBuffer_Release(&atb);if(pb.obj)PyBuffer_Release(&pb);if(wb.obj)PyBuffer_Release(&wb);if(kb.obj)PyBuffer_Release(&kb);return result;
}
static PyMethodDef methods[]={{"encode",encode,METH_VARARGS,"Learned generic scopes and deployment costs"},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_learned_scope_stage2",NULL,-1,methods};
PyMODINIT_FUNC PyInit__learned_scope_stage2(void){return PyModule_Create(&module);}
