#include <Python.h>
#include <numpy/arrayobject.h>
#include "td_parser.h"

static void dump1d(const double* v, int n) {
    for (int i=0;i<n; ++i) {
        printf("%lf  ",v[i]);
    }
    printf("\n");
}

static void dump2d(const double* v, int n, int m) {
    for (int i=0;i<n; ++i) {
        dump1d(v + i*n, m);
    }
    printf("\n");
}

static void dump2d_LD(const double*v, int n) {
    for (int i=0;i<n; ++i) {
        dump1d(v, i+1);
        v += (i+1);
    }
    printf("\n");
}

static PyObject* td_parser_np(PyObject *self, PyObject *args) {
    PyArrayObject *quote, *trade;
    int sutc, eutc, barsec;
    double tick_size;
    char* out_csv_fn;
    if (!PyArg_ParseTuple(args, "OOiiids",
                &quote, &trade, &sutc, &eutc, &barsec, 
                &tick_size, &out_csv_fn)) {
        printf("cannot parse, got %d, %d, %d\n", sutc,eutc,barsec);
        return NULL;
    }

    if ((quote->nd != 2) || (quote->dimensions[1] != 5)) {
        PyErr_SetString(PyExc_ValueError, "quote must be shape [n,5]");
        return NULL;
    }
    int n = quote->dimensions[0];
    if ( (trade->nd != 2) || (trade->dimensions[1] != 3)) {
        PyErr_SetString(PyExc_ValueError, "trade must be shape [m,3]");
        return NULL;
    }

    if ( (eutc<=sutc+1) || (barsec<1) ) {
        PyErr_SetString(PyExc_ValueError, "sutc>eutc-1 and barsec>0");
        return NULL;
    }

    if ( (quote->flags&1) && (trade->flags&1) ) {
        md::TickData2Bar tp("", "", sutc, eutc, barsec, tick_size);
        tp.parseDoubleArray((double*)quote->data, 
                (double*)trade->data, 
                quote->dimensions[0],
                trade->dimensions[0],
                out_csv_fn);
        return PyFloat_FromDouble(0.0);
    }
    // 1-d numpy array can also be non-continous, with non-1 strides
    printf("deal with non-continous d0/d1/d2\n");
    return NULL;
};

static PyMethodDef TDParserMethods[] =
{
    {"td_parser_np", td_parser_np, METH_VARARGS,
        "tickdata parser"},
    {NULL,NULL,0,NULL}
};

#if PY_MAJOR_VERSION >= 3
/* medule initialization Python version 3 */
static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "td_parser_module", "Some documentation",
    -1,
    TDParserMethods
};

PyMODINIT_FUNC PyInit_td_parser_module_np(void) {
    PyObject *module;
    module = PyModule_Create(&cModPyDem);
    if (module == NULL) {
        return NULL;
    }
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return module;
}

#else
/* module init Python version 2 */
PyMODINIT_FUNC inittrace_module_np(void) {
    PyObject *module;
    module = Py_InitModule("td_parser_module_np", TDParserMethods);
    if (module==NULL) return;
    import_array();
    return;
}
#endif

