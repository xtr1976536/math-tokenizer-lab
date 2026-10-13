/* Generic learned primitive protection and learned paired-byte scope signals.
 * No fixed TeX commands, delimiter characters, or mathematical AST parser. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <math.h>
typedef struct {int rank,left,right,vl,vr,newid;} MergeEvent;
static int earlier(MergeEvent a,MergeEvent b){return a.rank<b.rank||(a.rank==b.rank&&a.left<b.left);}
static void push(MergeEvent*h,int*size,MergeEvent e){int i=(*size)++;while(i>0){int p=(i-1)/2;if(!earlier(e,h[p]))break;h[i]=h[p];i=p;}h[i]=e;}
static MergeEvent pop(MergeEvent*h,int*size){MergeEvent out=h[0],last=h[--*size];int i=0;while(2*i+1<*size){int j=2*i+1;if(j+1<*size&&earlier(h[j+1],h[j]))j++;if(!earlier(h[j],last))break;h[i]=h[j];i=j;}if(*size)h[i]=last;return out;}
static int find_pair(const int32_t*table,int size,int a,int b,int*newid){uint32_t key=(uint32_t)a*2654435761u^(uint32_t)b*2246822519u;int i=key&(size-1);for(int k=0;k<size;k++){const int32_t*r=table+4*i;if(r[0]<0)return -1;if(r[0]==a&&r[1]==b){*newid=r[2];return r[3];}i=(i+1)&(size-1);}return -1;}
static void offer(MergeEvent*h,int*size,int a,int b,const int*ids,const int*version,const int32_t*table,int table_size){if(a<0||b<0)return;int newid,rank=find_pair(table,table_size,ids[a],ids[b],&newid);if(rank>=0)push(h,size,(MergeEvent){rank,a,b,version[a],version[b],newid});}
static PyObject*encode(PyObject*self,PyObject*args){
 PyObject *text,*children,*terminal,*achildren,*aterminal,*pairs,*costs,*token_costs,*merge_table,*atom_ids;int maxlen,with_cost=0;double scope,tree;
 if(!PyArg_ParseTuple(args,"OOOOOOOOOOidd|p",&text,&children,&terminal,&achildren,&aterminal,&pairs,&costs,&token_costs,&merge_table,&atom_ids,&maxlen,&scope,&tree,&with_cost))return NULL;
 if(!PyUnicode_Check(text)||maxlen<1||scope<0||!isfinite(scope)||tree<0||!isfinite(tree)){PyErr_SetString(PyExc_ValueError,"invalid learned scope configuration");return NULL;}
 Py_ssize_t length;const uint8_t*raw=(const uint8_t*)PyUnicode_AsUTF8AndSize(text,&length);if(!raw)return NULL;
 if(length>1000000){PyErr_SetString(PyExc_ValueError,"million-byte input limit");return NULL;}int n=(int)length;
 Py_buffer cb={0},tb={0},acb={0},atb={0},pb={0},wb={0},kb={0},mb={0},ib={0};PyObject*result=NULL;
 if(PyObject_GetBuffer(children,&cb,PyBUF_SIMPLE)<0)return NULL;
 if(PyObject_GetBuffer(terminal,&tb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(achildren,&acb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(aterminal,&atb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(pairs,&pb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(costs,&wb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(token_costs,&kb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(merge_table,&mb,PyBUF_SIMPLE)<0)goto buffers;
 if(PyObject_GetBuffer(atom_ids,&ib,PyBUF_SIMPLE)<0)goto buffers;
 if(mb.len%16||ib.len%4||mb.len<16){PyErr_SetString(PyExc_ValueError,"invalid merge tables");goto buffers;}
 if(cb.len%1024||tb.len%4||cb.len/1024!=tb.len/4||acb.len%1024||atb.len%4||acb.len/1024!=atb.len/4||pb.len%8||wb.len!=257*257*8||kb.len%8){PyErr_SetString(PyExc_ValueError,"invalid buffers");goto buffers;}
 const int32_t*ch=cb.buf,*term=tb.buf,*ach=acb.buf,*at=atb.buf,*pair=pb.buf;
 const double*w=wb.buf,*kcost=kb.buf;int nodes=tb.len/4,anodes=atb.len/4,V=kb.len/8;
 uint8_t*allowed=PyMem_Malloc((size_t)n+1),*marked=PyMem_Calloc((size_t)n+1,1);
 int *prefix=PyMem_Calloc((size_t)n+2,4),*stack=PyMem_Malloc(((size_t)n+1)*4),*end=PyMem_Malloc(((size_t)n+1)*4),*ids=PyMem_Malloc(((size_t)n+1)*4);
 int *unitid=PyMem_Malloc(((size_t)n+1)*4),*unitstart=PyMem_Malloc(((size_t)n+1)*4),*unitend=PyMem_Malloc(((size_t)n+1)*4),*next=PyMem_Malloc(((size_t)n+1)*4),*previous=PyMem_Malloc(((size_t)n+1)*4),*version=PyMem_Calloc((size_t)n+1,4),*maxend=PyMem_Calloc((size_t)n+1,4),*minstart=PyMem_Malloc(((size_t)n+1)*4);
 MergeEvent*heap=PyMem_Malloc(((size_t)n*3+3)*sizeof(MergeEvent));
 double*best=PyMem_Malloc(((size_t)n+1)*8);
 if(!allowed||!marked||!prefix||!stack||!end||!ids||!best||!unitid||!unitstart||!unitend||!next||!previous||!version||!maxend||!minstart||!heap){PyErr_NoMemory();goto memory;}
 memset(allowed,1,(size_t)n+1);
 int units=0;const int32_t*primitive_ids=ib.buf;
 for(int a=0;a<n;){int node=0,last=a+1,primitive=-1;
  for(int b=a;b<n&&b-a<maxlen;b++){node=ach[(size_t)node*256+raw[b]];if(node<0)break;if(node>=anodes){PyErr_SetString(PyExc_ValueError,"invalid primitive trie");goto memory;}if(at[node]>=0){last=b+1;primitive=at[node];}}
  if(primitive<0||primitive>=ib.len/4){PyErr_SetString(PyExc_ValueError,"missing primitive mapping");goto memory;}
  unitid[units]=primitive_ids[primitive];unitstart[units]=a;unitend[units]=last;next[units]=units+1;previous[units]=units-1;units++;
  for(int b=a+1;b<last;b++)allowed[b]=0;a=last;
 }
 if(units)next[units-1]=-1;for(int i=0;i<=n;i++)minstart[i]=n+1;
 int heap_size=0,table_size=mb.len/16;const int32_t*table=mb.buf;
 if(table_size&(table_size-1)){PyErr_SetString(PyExc_ValueError,"power-of-two hash table required");goto memory;}
 for(int i=0;i+1<units;i++)offer(heap,&heap_size,i,i+1,unitid,version,table,table_size);
 while(heap_size){MergeEvent e=pop(heap,&heap_size);int l=e.left,r=e.right;
  if(next[l]!=r||version[l]!=e.vl||version[r]!=e.vr)continue;
  unitid[l]=e.newid;unitend[l]=unitend[r];version[l]++;version[r]++;int following=next[r];next[l]=following;next[r]=-2;if(following>=0)previous[following]=l;
  int a=unitstart[l],b=unitend[l];if(b>maxend[a])maxend[a]=b;if(a<minstart[b])minstart[b]=a;
  offer(heap,&heap_size,previous[l],l,unitid,version,table,table_size);offer(heap,&heap_size,l,following,unitid,version,table,table_size);
 }
 for(int p=0;p<pb.len/8;p++){int depth=0;
  for(int a=0;a<n;a++){if(allowed[a]&&allowed[a+1]&&raw[a]==pair[2*p])stack[depth++]=a;
   else if(allowed[a]&&allowed[a+1]&&raw[a]==pair[2*p+1]&&depth){int start=stack[--depth];if(a>start+1){marked[start+1]=1;marked[a]=1;}}}
 }
 for(int a=0;a<=n;a++)prefix[a+1]=prefix[a]+marked[a];best[n]=0;
 for(int a=n-1;a>=0;a--){best[a]=INFINITY;end[a]=-1;if(!allowed[a])continue;int node=0,mx=0,mn=n+1;
  for(int b=a;b<n&&b-a<maxlen;b++){node=ch[(size_t)node*256+raw[b]];if(node<0)break;if(node>=nodes){PyErr_SetString(PyExc_ValueError,"invalid trie");goto memory;}
   int id=term[node],e=b+1;if(id>=0&&allowed[e]){if(id>=V){PyErr_SetString(PyExc_ValueError,"invalid token");goto memory;}
    int c=(e>=2?raw[e-2]:256)*257+raw[e-1];double value=1.+kcost[id]+tree*(mx>e||mn<a)+scope*(prefix[e]-prefix[a+1])+(e<n?w[c]:0.)+best[e];
    if(value<best[a]-1e-12){best[a]=value;end[a]=e;ids[a]=id;}}
   int edge=b+1;if(maxend[edge]>mx)mx=maxend[edge];if(minstart[edge]<mn)mn=minstart[edge];
  }
 }
 int pos=0,count=0;while(pos<n){if(end[pos]<=pos){PyErr_SetString(PyExc_RuntimeError,"protected lattice disconnected");goto memory;}count++;pos=end[pos];}
 result=PyList_New(count);if(!result)goto memory;pos=0;
 for(int i=0;i<count;i++){PyObject*id=PyLong_FromLong(ids[pos]);if(!id){Py_CLEAR(result);goto memory;}PyList_SET_ITEM(result,i,id);pos=end[pos];}
 if(with_cost)result=Py_BuildValue("Nd",result,best[0]);
memory:
 PyMem_Free(allowed);PyMem_Free(marked);PyMem_Free(prefix);PyMem_Free(stack);PyMem_Free(end);PyMem_Free(ids);PyMem_Free(best);PyMem_Free(unitid);PyMem_Free(unitstart);PyMem_Free(unitend);PyMem_Free(next);PyMem_Free(previous);PyMem_Free(version);PyMem_Free(maxend);PyMem_Free(minstart);PyMem_Free(heap);
buffers:
 if(cb.obj)PyBuffer_Release(&cb);if(tb.obj)PyBuffer_Release(&tb);if(acb.obj)PyBuffer_Release(&acb);if(atb.obj)PyBuffer_Release(&atb);if(pb.obj)PyBuffer_Release(&pb);if(wb.obj)PyBuffer_Release(&wb);if(kb.obj)PyBuffer_Release(&kb);if(mb.obj)PyBuffer_Release(&mb);if(ib.obj)PyBuffer_Release(&ib);return result;
}
static PyMethodDef methods[]={{"encode",encode,METH_VARARGS,"Learned generic scopes and deployment costs"},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_merge_scope_stage2",NULL,-1,methods};
PyMODINIT_FUNC PyInit__merge_scope_stage2(void){return PyModule_Create(&module);}
