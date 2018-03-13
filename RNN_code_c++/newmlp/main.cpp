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

class layer{
public:
    int ns;
    int num_node;
    int samth;
    double **forward_weight;
    double **backward_weight;
    double **dfw;
    double **dbw;
    double **bsdfw;
    double **bsdbw;
    int set_nn(int m){
        num_node=m;
        return 0;
    }
    int set_samth(int m){
        samth=m;
        return 0;
    }
    int get_samth(){
        return samth;
    }
};

class Ilayer:public layer{
public:
    double **data;
    Ilayer(int ns,int nn){
        this->ns=ns;
        this->num_node=nn;
        ifstream ifile;
        ifile.open("data");
        data=new double*[ns];
        for(int i=0;i<ns;i++){
            data[i]=new double[nn];
            for(int j=0;j<nn;j++){
                ifile>>data[i][j];
            }
        }
    }

};
class Hlayer:public layer{
public:
    double *ym;
    double *yn;
    double *bias;
    double *dbias;
    double *bsdbias;
    Hlayer(int nn){
        this->num_node=nn;
        ym=new double[nn];
        yn=new double[nn];
        bias=new double[nn];
        dbias=new double[nn];
        bsdbias=new double[nn];
        for(int i=0;i<nn;++i){
            ym[i]=0;
            yn[i]=0;
            bias[i]=getrand();
            dbias[i]=0;
            bsdbias[i]=0;
        }
    }
};
class Olayer:public layer{
public:
    double *yflag;
    double *yout;
    int label;
    int *all_label;
    double *error;
    int ncor;
    Olayer(int ns,int nn){
        this->ns=ns;
        this->num_node=nn;
        yflag=new double[nn];
        yout=new double[nn];
        error=new double[nn];
        all_label=new int[ns];
        ncor=0;
        ifstream ifile;
        ifile.open("label");
        for(int i=0;i<ns;++i){
            ifile>>all_label[i];
        }
        ifile.close();
        for(int i=0;i<nn;++i){
            yflag[i]=0;
            yout[i]=0;
            error[i]=0;
        }
        label=all_label[samth];
        yflag[label]=1;
    }
};

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
void init_net(Ilayer *input,Hlayer **hidden,Olayer *output){
    connection(input,hidden[0]);
    connection(hidden[0],hidden[1]);
    connection(hidden[1],output);
}
void cal_data(double *data,double **weight,int m,int n,double *bias,double *ym,double *yn){
    for(int i=0;i<m;i++){
        ym[i]=0;
        for(int j=0;j<n;j++){
            ym[i]+=weight[i][j]*data[j];
        }
        ym[i]+=bias[i];
        yn[i]=sigmoid(ym[i]);
    }
}
void cal_dataHO(double *data,double **weight,int m,int n,double *ym){
    for(int i=0;i<m;i++){
        ym[i]=0;
        for(int j=0;j<n;j++){
            ym[i]+=data[j]*weight[i][j];
        }
    }
}
void cal_error(double *y,double *error,double *yflag,int m,int label,int ncor,bool test){
    double max=-9999;
    int flag=0;
    double all=0;
    for(int i=0;i<m;i++){
        all+=exp(y[i]);
    }
    for(int i=0;i<m;++i){
        y[i]=(exp(y[i]))/all;
    }
    for(int i=0;i<m;++i){
        if(max<y[i]){
            max=y[i];
            flag=i;
        }
        error[i]=y[i]-yflag[i];
    }
    if(test){
        if(flag==label){
            ncor+=1;
        }
    }
}

void layer_forward(Ilayer *input,Hlayer **hidden,Olayer *output,int m){
    double *ptr=*(input->data+m);
    cal_data(ptr,input->forward_weight,hidden[0]->num_node,input->num_node,hidden[0]->bias,hidden[0]->ym,hidden[0]->yn);
    cal_data(hidden[0]->yn,hidden[0]->forward_weight,hidden[1]->num_node,hidden[0]->num_node,hidden[1]->bias,hidden[1]->ym,hidden[0]->yn);
    cal_dataHO(hidden[1]->yn,hidden[1]->forward_weight,output->num_node,hidden[1]->num_node,output->yout);
    cal_error(output->yout,output->error,output->yflag,output->num_node,output->label,output->ncor,);
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    int epoch=10000;
    int batch=20;
    double lr=0.001;
    int ns=500;//yang ben shuliang
    int input_num_node=30;
    bool test=false;
    int test_internal=10;
    int num_hiddenlayer=2;
    int num_hiddenlayer_node[num_hiddenlayer];
    for(int i=0;i<num_hiddenlayer;i++){
        num_hiddenlayer_node[i]=50;
    }
    Ilayer input(ns,input_num_node);
    Hlayer *hidden[num_hiddenlayer];
    for(int i=0;i<num_hiddenlayer;i++){
        hidden[i]=new Hlayer(num_hiddenlayer_node[i]);
    }
    Olayer output(ns,2);
    for(int i=0;i<epoch;++i){
        if((i+1)%test_internal==0){
            test=true;
        }
    }
    return a.exec();
}
