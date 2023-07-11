#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

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

static double tic(int n, double* d0, double* d1, double*d2) {
    //time_t t1 = time(0);

    // calculate the trace inverse of banded Z, 
    // with diag,sub,subsub diag as d0,d1,d2
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
    double* dd1 = (double*)malloc(sizeof(double)*(n-1));
    dd0[0] = d0[0];
    dd1[0] = d1[0];

    double* ff0 = (double*)malloc(sizeof(double)*n);
    double* ff1 = (double*)malloc(sizeof(double)*(n-1));
    int i=0;
    for (i=0; i<n-3; ++i) {
        //f = E[0,1:]/E[0,0]
        register const double f0 = (*e01)/(*e00), f1 = (*e02)/(*e00);

        //E[0,:2] = E[1,1:]-f*E[0,1]
        *e00 = (*e11)-f0*(*e01);
        *e01 = (*e12)-f1*(*e01);

        //E[1,:2] = E[2,1:]-f*E[0,2]
        *e10 = (*e21)-f0*(*e02);
        *e11 = (*e22)-f1*(*e02);

        //E[2,:] = np.array([d2[i+1],d1[i+2],d0[i+3]])
        *e20 = d2[i+1];
        *e21 = d1[i+2];
        *e22 = d0[i+3];

        // E[:2,2]= np.array([d2[i+1],d1[i+2]])
        *e02 = d2[i+1];
        *e12 = d1[i+2];

        // dd.append(E[0,:2].copy())
        dd0[i+1] = *e00;
        dd1[i+1] = *e01;

        //ff.append(f)
        ff0[i] = f0;
        ff1[i] = f1;
    }

    // the last one
    i = n-3;

    //f = E[0,1:]/E[0,0]
    register double f0 = (*e01)/(*e00), f1 = (*e02)/(*e00);

    // E[0,:2] = E[1,1:]-f*E[0,1]
    *e00 = (*e11)-f0*(*e01);
    *e01 = (*e12)-f1*(*e01);

    //E[1,:2] = E[2,1:]-f*E[0,2]
    *e10 = (*e21)-f0*(*e02);
    *e11 = (*e22)-f1*(*e02);

    // dd.append(E[0,:2].copy())
    dd0[i+1] = *e00;
    dd1[i+1] = *e01;
    
    // ff.append(f)
    ff0[i] = f0;
    ff1[i] = f1;

    i = n-2;

    //f = E[0,1:]/E[0,0]
    f0 = (*e01)/(*e00), f1 = (*e02)/(*e00);

    //E[0,:2] = E[1,1:]-f*E[0,1]
    *e00 = (*e11)-f0*(*e01);
    *e01 = (*e12)-f1*(*e01);
    
    //dd.append(E[0,:2].copy())
    dd0[i+1] = *e00;
    dd1[i+1] = *e01;
    
    //ff.append(f)
    ff0[i] = f0;
    ff1[i] = f1;

    //# create a big eye
    //L1 = np.eye(n)
    
    double* L1p = (double*)malloc(sizeof(double)*n*(n+1)/2);
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

    //time_t t2 = time(0);
    printf("done L1, %d\n", (int)(t2-t1));

    //return dd[:,0], dd[:-1,1], d2, L1

    // now run it again with dd0, dd1, and d2
    //dd1 = d1/d0[:-1]
    //dd2 = d2/d0[:-2]
    //# create a big eye
    //L2 = np.eye(n)

    double* L2p = (double*)malloc(sizeof(double)*n*(n+1)/2);
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

    //time_t t3 = time(0);
    printf("done L2, %d\n", (int)(t3-t2));

    // trace [ L1.T, dd0^{-1}, L2 ]
    register double sum = 0.0;
    const double *bx = dd0;
    for (i=0; i<n; ++i, ++bx) {
        register const double *l1 = L1[i], *l2 = L2[i];
        const double bxv = *bx;
        for (int j=0; j<=i; ++j, ++l1, ++l2) {
            sum += ((*l1) * (*l2) / bxv);
        }
    }

    //time_t t4 = time(0);
    //printf("done L1 B L2, %d\n", (int)(t4-t3));
    /*
    dump2d_LD(L1p,n);
    dump2d_LD(L2p,n);
    dump1d(dd0,n);
    */
    return sum;
}

double test() {
    int n = 6;
    double d0[] = { 1.0, 2.0, 3.0, 4.0, -1.0, -9 };
    double d1[] = { 5.0, 6.0, -7.0, 2.0, 2 };
    double d2[] = { -8.0, 9.0, 1.0, 7};

    double sum = tic(n, d0, d1, d2);
    double diff = -0.38844133099824874;
    printf("sum = %lf, diff = %lf\n", sum, diff);
    return diff;
}

void gen_d0_d1_d2(int n, double**d0_, double**d1_, double**d2_) {
    double* d0 = (double*)malloc(sizeof(double)*n);
    double* d1 = (double*)malloc(sizeof(double)*(n-1));
    double* d2 = (double*)malloc(sizeof(double)*(n-2));
    int X = 1709;
    for (int i = 0; i<n-2; ++i) {
        d0[i] = (i%X+1) * 0.1 * (1+i/X);
        d1[i] = ((i+7)%X+1) * 0.1 * (1+i/X);
        d2[i] = ((i+17)%X+1) * 0.1 * (1+i/X);
    }
    d0[n-2] = 171.2;
    d1[n-2] = 172.2;
    d0[n-1] = 173.2;

    *d0_=d0;
    *d1_=d1;
    *d2_=d2;

    /*
    dump1d(d0,n);
    dump1d(d1,n-1);
    dump1d(d2,n-2);
    */
}

int main(int argc, char**argv) {
    /*
    test();
    return 0;
    */

    int n = atoi(argv[1]);
    double *d0, *d1, *d2;

    gen_d0_d1_d2(n, &d0, &d1, &d2);

    time_t t0 = time(0);
    double sum = tic(n, d0, d1, d2);
    time_t t1 = time(0);

    printf("%lf\n%d\n",sum, (int)(t1-t0));
    return 0;
}
