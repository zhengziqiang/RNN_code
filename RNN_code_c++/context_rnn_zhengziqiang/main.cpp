#include <QCoreApplication>
#include<vector>
#include<time.h>
#include<fstream>
#include<string>
using namespace std;
double getrand(){
    static double V1, V2, S;
    static int phase = 0;
    double X;
    if (phase == 0) {
        do {
            double U1 = (double) rand() / RAND_MAX;
            double U2 = (double) rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    phase = 1 - phase;
    double x=X*0.1;
    return x;
}
//sigmoid函数
inline double sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}
//sigmoid的导函数
inline double fsigmoid(double x){
    return x*(1-x);
}

void connection(layer *l1,layer *l2){
    int m=l1->num_node;
    int n=l2->num_node;
    l1->forward_weight=new double*[n];
    l1->bsdfw=new double*[n];
    l1->dfw=new double*[n];
    for(int i=0;i<n;++i){
        l1->forward_weight[i]=new double[m];
        l1->bsdfw[i]=new double[m];
        l1->dfw[i]=new double[m];
        for(int j=0;j<m;++j){
            l1->forward_weight[i][j]=getrand();
            l1->bsdfw[i][j]=0;
            l1->dfw[i][j]=0;
        }
    }
    l2->backward_weight=new double*[m];
    l2->bsdbw=new double *[m];
    l2->dbw=new double*[m];
    for(int i=0;i<m;++i){
        l2->backward_weight[i]=new double[n];
        l2->dbw[i]=new double[n];
        l2->bsdbw[i]=new double[n];
        for(int j=0;j<n;j++){
            l2->backward_weight[i][j]=l1->forward_weight[j][i];
            l2->bsdbw[i][j]=0;
            l2->dbw[i][j]=0;
        }
    }
}
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    return a.exec();
}
