/* Byte-context signal + dictionary DP. No mathematical parser or rules. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <math.h>
#define CONTEXTS (257*257)

static PyObject *encode(PyObject *self, PyObject *args) {
    PyObject *text, *children, *terminal, *costs;
    int maxlen, with_cost=0;
    if (!PyArg_ParseTuple(args,"OOOOi|p",&text,&children,&terminal,&costs,&maxlen,&with_cost)) return NULL;
    if (!PyUnicode_Check(text) || maxlen<1) {PyErr_SetString(PyExc_ValueError,"invalid input");return NULL;}
    Py_ssize_t length;
    const unsigned char *raw=(const unsigned char*)PyUnicode_AsUTF8AndSize(text,&length);
    if (!raw) return NULL;
    if (length>1000000) {PyErr_SetString(PyExc_ValueError,"million-byte input limit");return NULL;}
    Py_buffer cb={0},tb={0},wb={0}; PyObject *result=NULL;
    if (PyObject_GetBuffer(children,&cb,PyBUF_SIMPLE)<0) return NULL;
    if (PyObject_GetBuffer(terminal,&tb,PyBUF_SIMPLE)<0) goto buffers;
    if (PyObject_GetBuffer(costs,&wb,PyBUF_SIMPLE)<0) goto buffers;
    if (cb.len%1024 || tb.len%4 || cb.len/1024!=tb.len/4 || wb.len!=CONTEXTS*8) {
        PyErr_SetString(PyExc_ValueError,"invalid buffers");goto buffers;
    }
    const int32_t *ch=cb.buf,*term=tb.buf;const double *w=wb.buf;
    int nodes=(int)(tb.len/4),n=(int)length;
    double *best=PyMem_Malloc(((size_t)n+1)*sizeof(double));
    int *end=PyMem_Malloc(((size_t)n+1)*sizeof(int)),*ids=PyMem_Malloc(((size_t)n+1)*sizeof(int));
    if (!best || !end || !ids) {PyErr_NoMemory();goto memory;}
    best[n]=0;
    for (int a=n-1;a>=0;a--) {
        best[a]=INFINITY;end[a]=-1;int node=0;
        for (int b=a;b<n && b-a<maxlen;b++) {
            node=ch[(size_t)node*256+raw[b]];
            if (node<0) break;
            if (node>=nodes) {PyErr_SetString(PyExc_ValueError,"invalid trie child");goto memory;}
            int id=term[node],e=b+1;
            if (id>=0) {
                int c=(e>=2 ? raw[e-2] : 256)*257+raw[e-1];
                double value=1.+(e<n ? w[c] : 0.)+best[e];
                if (value<best[a]-1e-12) {best[a]=value;end[a]=e;ids[a]=id;}
            }
        }
    }
    int count=0,pos=0;
    while (pos<n) {
        if (end[pos]<=pos) {PyErr_SetString(PyExc_RuntimeError,"missing byte fallback");goto memory;}
        count++;pos=end[pos];
    }
    result=PyList_New(count);if (!result) goto memory;pos=0;
    for (int i=0;i<count;i++) {
        PyObject *id=PyLong_FromLong(ids[pos]);if (!id) {Py_CLEAR(result);goto memory;}
        PyList_SET_ITEM(result,i,id);pos=end[pos];
    }
    if (with_cost) result=Py_BuildValue("Nd",result,best[0]);
memory:
    PyMem_Free(best);PyMem_Free(end);PyMem_Free(ids);
buffers:
    if (cb.obj) PyBuffer_Release(&cb);
    if (tb.obj) PyBuffer_Release(&tb);
    if (wb.obj) PyBuffer_Release(&wb);
    return result;
}
static PyMethodDef methods[]={{"encode",encode,METH_VARARGS,"Fused byte-signal segmentation"},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_signal_stage2",NULL,-1,methods};
PyMODINIT_FUNC PyInit__signal_stage2(void) {return PyModule_Create(&module);}
