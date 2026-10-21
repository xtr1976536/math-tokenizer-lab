/* Ranked BPE with hard source-tree interval compatibility. The mathematical
 * frontend is shared with AST-v4; every merge is checked before execution. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "frontend_v6.c"
typedef struct {int rank,l,r,vl,vr,id;} Event;
static int before(Event a,Event b){return a.rank<b.rank||(a.rank==b.rank&&a.l<b.l);}
static void push(Event*h,int*z,Event e){int i=(*z)++;while(i){int p=(i-1)/2;if(!before(e,h[p]))break;h[i]=h[p];i=p;}h[i]=e;}
static Event pop(Event*h,int*z){Event e=h[0],v=h[--*z];int i=0;while(2*i+1<*z){int j=2*i+1;if(j+1<*z&&before(h[j+1],h[j]))j++;if(!before(h[j],v))break;h[i]=h[j];i=j;}if(*z)h[i]=v;return e;}
static void offer(Event*h,int*z,int l,int r,int*id,int*v,const int32_t*t,int size){
 if(l<0||r<0)return;uint32_t key=(uint32_t)id[l]*2654435761u^(uint32_t)id[r]*2246822519u;int i=key&(size-1);
 for(int k=0;k<size;k++){const int32_t*p=t+4*i;if(p[0]<0)return;if(p[0]==id[l]&&p[1]==id[r]){push(h,z,(Event){p[3],l,r,v[l],v[r],p[2]});return;}i=(i+1)&(size-1);}
}
static PyObject*encode(PyObject*self,PyObject*args){
 PyObject*text,*table;if(!PyArg_ParseTuple(args,"OO",&text,&table))return NULL;
 Py_ssize_t length;const uint8_t*raw=(const uint8_t*)PyUnicode_AsUTF8AndSize(text,&length);if(!raw)return NULL;
 if(length>1000000){PyErr_SetString(PyExc_ValueError,"input exceeds one million bytes");return NULL;}
 Py_buffer b={0};if(PyObject_GetBuffer(table,&b,PyBUF_SIMPLE)<0)return NULL;
 int size=b.len/16,n=(int)length;if(b.len%16||!size||(size&(size-1))){PyBuffer_Release(&b);PyErr_SetString(PyExc_ValueError,"invalid merge table");return NULL;}
 int *id=PyMem_Malloc((n+1)*4),*end=PyMem_Malloc((n+1)*4),*next=PyMem_Malloc((n+1)*4),*prev=PyMem_Malloc((n+1)*4),*v=PyMem_Calloc(n+1,4),*mx=PyMem_Calloc(n+1,4),*mn=PyMem_Malloc((n+1)*4);
 int32_t*spans=PyMem_Malloc((8*(size_t)n+1)*8);Event*heap=PyMem_Malloc((3*(size_t)n+3)*sizeof(Event));PyObject*out=NULL;
 if(!id||!end||!next||!prev||!v||!mx||!mn||!spans||!heap){PyErr_NoMemory();goto done;}
 for(int i=0;i<n;i++){id[i]=raw[i];end[i]=i+1;next[i]=i+1;prev[i]=i-1;}if(n)next[n-1]=-1;
 for(int i=0;i<=n;i++)mn[i]=n+1;
 int ns=mt_tree(raw,n,spans);if(ns<0){PyErr_NoMemory();goto done;}
 for(int i=0;i<ns;i++){int a=spans[2*i],e=spans[2*i+1];if(e>mx[a])mx[a]=e;if(a<mn[e])mn[e]=a;}
 int z=0;for(int i=0;i+1<n;i++)offer(heap,&z,i,i+1,id,v,b.buf,size);
 while(z){Event e=pop(heap,&z);int l=e.l,r=e.r;if(next[l]!=r||v[l]!=e.vl||v[r]!=e.vr)continue;
  int legal=1;for(int p=l+1;p<end[r];p++)if(mx[p]>end[r]||mn[p]<l){legal=0;break;}if(!legal)continue;
  id[l]=e.id;end[l]=end[r];v[l]++;v[r]++;next[l]=next[r];next[r]=-2;if(next[l]>=0)prev[next[l]]=l;
  offer(heap,&z,prev[l],l,id,v,b.buf,size);offer(heap,&z,l,next[l],id,v,b.buf,size);
 }
 int count=0;for(int i=0;i<n;i++)count+=next[i]!=-2;out=PyList_New(count);if(!out)goto done;
 for(int i=0,j=0;i<n;i++)if(next[i]!=-2){PyObject*t=PyLong_FromLong(id[i]);if(!t){Py_CLEAR(out);goto done;}PyList_SET_ITEM(out,j++,t);}
done:
 PyMem_Free(id);PyMem_Free(end);PyMem_Free(next);PyMem_Free(prev);PyMem_Free(v);PyMem_Free(mx);PyMem_Free(mn);PyMem_Free(spans);PyMem_Free(heap);PyBuffer_Release(&b);return out;
}
static PyMethodDef methods[]={{"encode",encode,METH_VARARGS,"Hard AST ranked byte merges"},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_hard_ranked_2026",NULL,-1,methods};
PyMODINIT_FUNC PyInit__hard_ranked_2026(void){return PyModule_Create(&module);}
