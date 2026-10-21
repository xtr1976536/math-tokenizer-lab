/* Python bridge, exact same algorithms as the ctypes reference. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "frontend_v6.c"
#include "compact_codec_v6.c"
#include "decoder_v4.c"

static int buffers(PyObject *text,PyObject *children,PyObject *terminal,const uint8_t **raw,Py_ssize_t *n,Py_buffer *cb,Py_buffer *tb){
    if(!PyUnicode_Check(text)){PyErr_SetString(PyExc_TypeError,"text must be str");return -1;}
    *raw=(const uint8_t*)PyUnicode_AsUTF8AndSize(text,n);if(!*raw)return -1;
    if(*n>INT32_MAX/16){PyErr_SetString(PyExc_ValueError,"input too large");return -1;}
    if(PyObject_GetBuffer(children,cb,PyBUF_SIMPLE)<0)return -1;
    if(PyObject_GetBuffer(terminal,tb,PyBUF_SIMPLE)<0){PyBuffer_Release(cb);return -1;}
    if(cb->len<256*4 || tb->len<4 || cb->len%1024 || cb->len/1024!=tb->len/4){
        PyBuffer_Release(cb);PyBuffer_Release(tb);PyErr_SetString(PyExc_ValueError,"invalid trie buffers");return -1;
    }
    return 0;
}

static PyObject *symbols(PyObject *self,PyObject *args){
    PyObject *text,*children,*terminal;if(!PyArg_ParseTuple(args,"OOO",&text,&children,&terminal))return NULL;
    const uint8_t *raw;Py_ssize_t n;Py_buffer cb,tb;
    if(buffers(text,children,terminal,&raw,&n,&cb,&tb)<0)return NULL;
    uint8_t *out=PyMem_Malloc((size_t)n*3+1);PyObject *result=NULL;
    if(!out)PyErr_NoMemory();else{
        int count=mt_symbols_compact(raw,(int)n,cb.buf,tb.buf,out);
        result=PyUnicode_DecodeUTF8((const char*)out,count,"strict");PyMem_Free(out);
    }
    PyBuffer_Release(&cb);PyBuffer_Release(&tb);return result;
}

static PyObject *ast_encode(PyObject *self,PyObject *args){
    PyObject *text,*children,*terminal;int maxlen,hard;
    if(!PyArg_ParseTuple(args,"OOOii",&text,&children,&terminal,&maxlen,&hard))return NULL;
    const uint8_t *raw;Py_ssize_t n;Py_buffer cb,tb;
    if(buffers(text,children,terminal,&raw,&n,&cb,&tb)<0)return NULL;
    int32_t *spans=PyMem_Malloc(((size_t)n*16+2)*sizeof(int32_t));
    int32_t *out=PyMem_Malloc(((size_t)n+1)*sizeof(int32_t));uint8_t *allowed=PyMem_Malloc((size_t)n+1);
    PyObject *result=NULL;
    if(!spans||!out||!allowed)PyErr_NoMemory();else{
        int ns=mt_tree(raw,(int)n,spans);
        if(ns<0)PyErr_NoMemory();else{
            mt_allowed(raw,(int)n,cb.buf,tb.buf,allowed);
            int count=mt_shortest(raw,(int)n,cb.buf,tb.buf,maxlen,spans,ns,hard,allowed,out);
            if(count<0)PyErr_SetString(PyExc_RuntimeError,"AST constraint graph disconnected");else{
                result=PyList_New(count);
                if(result)for(int i=0;i<count;i++){
                    PyObject *id=PyLong_FromLong(out[i]);if(!id){Py_CLEAR(result);break;}PyList_SET_ITEM(result,i,id);
                }
            }
        }
    }
    PyMem_Free(spans);PyMem_Free(out);PyMem_Free(allowed);PyBuffer_Release(&cb);PyBuffer_Release(&tb);return result;
}

static PyObject *wordpiece(PyObject*self,PyObject*args){
    PyObject *data,*fc,*ft,*cc,*ct;
    if(!PyArg_ParseTuple(args,"OOOOO",&data,&fc,&ft,&cc,&ct))return NULL;
    Py_buffer raw,fcb,ftb,ccb,ctb;
    if(PyObject_GetBuffer(data,&raw,PyBUF_SIMPLE)<0)return NULL;
    if(PyObject_GetBuffer(fc,&fcb,PyBUF_SIMPLE)<0){PyBuffer_Release(&raw);return NULL;}
    if(PyObject_GetBuffer(ft,&ftb,PyBUF_SIMPLE)<0){PyBuffer_Release(&raw);PyBuffer_Release(&fcb);return NULL;}
    if(PyObject_GetBuffer(cc,&ccb,PyBUF_SIMPLE)<0){PyBuffer_Release(&raw);PyBuffer_Release(&fcb);PyBuffer_Release(&ftb);return NULL;}
    if(PyObject_GetBuffer(ct,&ctb,PyBUF_SIMPLE)<0){PyBuffer_Release(&raw);PyBuffer_Release(&fcb);PyBuffer_Release(&ftb);PyBuffer_Release(&ccb);return NULL;}
    int n=(int)raw.len,pos=0;uint8_t*d=raw.buf;PyObject*result=PyList_New(0);
    while(result&&pos<n){
        int32_t *children=pos?(int32_t*)ccb.buf:(int32_t*)fcb.buf,*terminal=pos?(int32_t*)ctb.buf:(int32_t*)ftb.buf;
        int node=0,best=-1,end=-1;
        for(int b=pos;b<n;b++){
            node=children[(size_t)node*256+d[b]];if(node<0)break;
            if(terminal[node]>=0){best=terminal[node];end=b+1;}
        }
        if(best<0){Py_CLEAR(result);Py_INCREF(Py_None);result=Py_None;break;}
        PyObject*id=PyLong_FromLong(best);
        if(!id||PyList_Append(result,id)<0){Py_XDECREF(id);Py_CLEAR(result);break;}Py_DECREF(id);pos=end;
    }
    PyBuffer_Release(&raw);PyBuffer_Release(&fcb);PyBuffer_Release(&ftb);PyBuffer_Release(&ccb);PyBuffer_Release(&ctb);return result;
}

static PyMethodDef methods[]={{"symbols",symbols,METH_VARARGS,"Exact native lexical alphabet transform."},{"ast_encode",ast_encode,METH_VARARGS,"Exact native binding tree and constrained shortest path."},{"wordpiece",wordpiece,METH_VARARGS,"Equivalent longest-match WordPiece over reversible byte alphabet."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_math_frontend_v6",NULL,-1,methods};
PyMODINIT_FUNC PyInit__math_frontend_v6(void){return PyModule_Create(&module);}
