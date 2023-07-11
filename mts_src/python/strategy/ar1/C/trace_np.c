#include <Python.h>
#include <numpy/arrayobject.h>

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

static double c_sym_penta(int n, double* d0, double* d1, double*d2) {
    double E[3][3] = {0};
    register double *e00,*e01,*e02,*e10,*e11,*e12,*e20,*e21,*e22;
    e00 = &(E[0][0]);
    e01 = &(E[0][1]);
    e02 = &(E[0][2]);
    e10 = &(E[1][0]);
    e11 = &(E[1][1]);
    e12 = &(E[1][2]);
    e20 = &(E[2][0]);
    e21 = &(E[2][1]);
    e22 = &(E[2][2]);

    E[0][0]=d0[0];
    E[1][1]=d0[1];
    E[2][2]=d0[2];
    E[0][1]=d1[0];
    E[1][0]=d1[0];
    E[1][2]=d1[1];
    E[2][1]=d1[1];
    E[2][0]=d2[0];
    E[0][2]=d2[0];
    double* dd0 = (double*)malloc(sizeof(double)*n);
    double* dd1 = (double*)malloc(sizeof(double)*n);

    ////////////////////////
    // debug
    //memset(dd0, 0, sizeof(double)*n);
    //memset(dd1, 0, sizeof(double)*n);
    //free (dd0);
    //free (dd1);
    //return 0.0;
    ////////////////////////


    dd0[0] = d0[0];
    dd1[0] = d1[0];

    double* ff0 = (double*)malloc(sizeof(double)*n);
    double* ff1 = (double*)malloc(sizeof(double)*n);
    int i=0;
    for (i=0; i<n-3; ++i) {
        register const double f0 = (*e01)/(*e00), f1 = (*e02)/(*e00);
        *e00 = (*e11)-f0*(*e01);
        *e01 = (*e12)-f1*(*e01);
        *e10 = (*e21)-f0*(*e02);
        *e11 = (*e22)-f1*(*e02);
        *e20 = d2[i+1];
        *e21 = d1[i+2];
        *e22 = d0[i+3];
        *e02 = d2[i+1];
        *e12 = d1[i+2];
        dd0[i+1] = *e00;
        dd1[i+1] = *e01;
        ff0[i] = f0;
        ff1[i] = f1;
    }
    i = n-3;
    register double f0 = (*e01)/(*e00), f1 = (*e02)/(*e00);
    *e00 = (*e11)-f0*(*e01);
    *e01 = (*e12)-f1*(*e01);
    *e10 = (*e21)-f0*(*e02);
    *e11 = (*e22)-f1*(*e02);
    dd0[i+1] = *e00;
    dd1[i+1] = *e01;
    ff0[i] = f0;
    ff1[i] = f1;
    i = n-2;
    f0 = (*e01)/(*e00), f1 = (*e02)/(*e00);
    *e00 = (*e11)-f0*(*e01);
    *e01 = (*e12)-f1*(*e01);
    dd0[i+1] = *e00;
    dd1[i+1] = *e01;
    ff0[i] = f0;
    ff1[i] = f1;

    ////////////////////////
    // debug
    //memset(ff0, 0, sizeof(double)*n);
    //memset(ff1, 0, sizeof(double)*n);
    //free (dd0);
    //free (dd1);
    //free (ff0);
    //free (ff1);
    //printf("DONE!\n");
    //return 0.0;
    ////////////////////////
    //

    double* L1p = (double*)malloc(sizeof(double)*n*(n+1)/2);
    memset(L1p, 0, sizeof(double)*n*(n+1)/2);
    double** L1 = (double**)malloc(sizeof(double*)*n);
    double* p = L1p;
    for (i=0;i<n;++i) {
        L1[i] = p;
        p[i] = 1.0;
        p += (i+1);
    }
    for (i=0; i<n-2;++i) {
        double *p0 = L1[i], *p1 = L1[i+1], *p2=L1[i+2];
        const double ff0_i = ff0[i], ff1_i=ff1[i];
        for (int j=0; j<=i;++j, ++p0, ++p1, ++p2) {
            *p1 -= ((*p0) * ff0_i);
            *p2 -= ((*p0) * ff1_i);
        }
    }
    i = n-2;
    double *p0 = L1[i], *p1 = L1[i+1];
    double ff0_i = ff0[i];
    for (int j=0; j<=i; ++j, ++p0, ++p1) {
        *p1 -= ( (*p0) * ff0_i );
    };

    double* L2p = (double*)malloc(sizeof(double)*n*(n+1)/2);
    memset(L2p, 0, sizeof(double)*n*(n+1)/2);
    double** L2 = (double**)malloc(sizeof(double*)*n);
    p = L2p;
    for (i=0;i<n;++i) {
        dd1[i] /= dd0[i];
        L2[i] = p;
        p[i] = 1.0;
        p += (i+1);
    }
    for (i=0; i<n-2;++i) {
        double *p0 = L2[i], *p1 = L2[i+1], *p2=L2[i+2];
        const double ff0_i = dd1[i],  ff1_i=d2[i]/dd0[i];
        for (int j=0; j<=i;++j, ++p0, ++p1, ++p2) {
            *p1 -= ((*p0) * ff0_i);
            *p2 -= ((*p0) * ff1_i);
        }
    }
    i = n-2;
    p0 = L2[i];
    p1 = L2[i+1];
    ff0_i = dd1[i];
    for (int j=0; j<=i; ++j, ++p0, ++p1) {
        *p1 -= ( (*p0) * ff0_i );
    };

    register double sum = 0.0;
    const double *bx = dd0;
    for (i=0; i<n; ++i, ++bx) {
        register const double *l1 = L1[i], *l2 = L2[i];
        const double bxv = *bx;
        for (int j=0; j<=i; ++j, ++l1, ++l2) {
            sum += ((*l1) * (*l2) / bxv);
        }
    }
    /*
    dump2d_LD(L1p,n);
    dump2d_LD(L2p,n);
    dump1d(dd0,n);
    */
    free(dd0);
    free(dd1);
    free(ff0);
    free(ff1);
    free(L1p);
    free(L1);
    free(L2p);
    free(L2);
    return sum;
}

static double c_sym_penta2(int n, double* d0, double* d1, double*d2) {
    return 0.0;
}

static PyObject* trace_np(PyObject *self, PyObject *args) {
    PyArrayObject *d0, *d1, *d2;
    if (!PyArg_ParseTuple(args, "OOO",
                &d0, &d1, &d2)) {
        return NULL;
    }

    if ((d0->nd != 1) || (d0->descr->type_num != PyArray_DOUBLE)) {
        PyErr_SetString(PyExc_ValueError, "d0 must be 1-dim with type float");
        return NULL;
    }

    int n = d0->dimensions[0];
    if ( (d1->nd != 1) || (d1->dimensions[0] != n-1) ) {
        PyErr_SetString(PyExc_ValueError, "d1 must be 1-dim with size 1-less d0");
        return NULL;
    }
    if ( (d2->nd != 1) || (d2->dimensions[0] != n-2) ) {
        PyErr_SetString(PyExc_ValueError, "d2 must be 1-dim with size 2-less d0");
        return NULL;
    }

    if ( (d0->flags&1) && (d1->flags&1) && (d2->flags&1)) {
        double v = c_sym_penta(n, (double*)d0->data, (double*)d1->data, (double*)d2->data);
        return PyFloat_FromDouble(v);
    }

    // 1-d numpy array can also be non-continous, with non-1 strides
    printf("deal with non-continous d0/d1/d2\n");
    return NULL;
};

static PyMethodDef TraceMethods[] =
{
    {"trace_np", trace_np, METH_VARARGS,
        "evaluate a square matrix trace"},
    {NULL,NULL,0,NULL}
};

#if PY_MAJOR_VERSION >= 3
/* medule initialization Python version 3 */
static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "trace_module", "Some documentation",
    -1,
    TraceMethods
};

PyMODINIT_FUNC PyInit_trace_module_np(void) {
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
    module = Py_InitModule("trace_module_np", TraceMethods);
    if (module==NULL) return;
    import_array();
    return;
}
#endif

