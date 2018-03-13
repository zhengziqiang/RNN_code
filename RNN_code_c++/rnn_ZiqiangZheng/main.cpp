#include <QCoreApplication>
#include<iostream>
#include<fstream>
#include<vector>
#include<math.h>
#include<time.h>
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
inline double sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}
//sigmoid的导函数
inline double fsigmoid(double x){
    return x*(1-x);
}

void init(vector <double> &param,int m){
    for(int i=0;i<m;i++){
        param.push_back(0.0);
    }
}
void init_dw(vector <vector <double> > &dw,int m,int n){
    vector <vector <double> > param(m,vector<double>(n,0.0));
    dw=param;
}
void init(vector <vector <double> > &tth,int tt,int nn){
    vector <vector <double> > param(tt,vector<double>(nn,0.0));
    tth=param;
}
struct iparam{
    vector <vector <double> > data;
};

struct hparam{
    int nn;
    int tt;//time
    vector <vector <double> > dht;//delta_h
    vector <vector <double> > at;//no sigmoid_val
    vector <vector <double> > bt;//sigmoid_val
    vector <vector <double> > db;//delta_bias
    vector <double> dba;//delta_biasall
    vector <vector <double> > dwih;//delta_weight
    vector <vector <double> > dwaih;//delta_weightall
    vector <vector <double> > dwhh;
    vector <vector <double> > dwahh;
    vector <vector <double> > dwho;
    vector <vector <double> > dwaho;
    vector <vector <double> > bsdih;
    vector <vector <double> > bsdhh;
    vector <vector <double> > bsdho;
    vector <double> bsdb;
};

struct oparam{
    double errorall;
    vector <double> label;
    vector <vector <double> > op;
    vector <double> error;
};

class layer{
public:
    int nn;
    int bs;
    int tt;
    int set_nn(int m){
        nn=m;
        return 0;
    }
    int get_nn(){
        return nn;
    }
    int set_bs(int m){
        bs=m;
        return 0;
    }
    int get_bs(){
        return bs;
    }
    int set_tt(int m){
        tt=m;
        return 0;
    }
};

class Ilayer:public layer{
public:
    iparam in_param;
    void ilayer(int nn,int tt,struct iparam &in_param);
};
void Ilayer::ilayer(int nn,int tt,iparam &in_param){
    init(in_param.data,tt,nn);
}

class Hlayer:public layer{
public:
    int count;
    int zero_count(){
        count=0;
        return 0;
    }
    int countjia(){
        count+=1;
        return 0;
    }
    hparam hi_param;
    void hlayer(int nn,int tt,struct hparam &hi_param);
};
void Hlayer::hlayer(int nn,int tt, hparam &hi_param){
    hi_param.nn=nn;
    hi_param.tt=tt;
    init(hi_param.db,tt,nn);
    init(hi_param.dba,nn);
    init(hi_param.dht,tt,nn);
    init(hi_param.at,tt,nn);
    init(hi_param.bt,tt,nn);
    init(hi_param.bsdb,nn);
}

class Olayer:public layer{
public:
    oparam ou_param;
    int lth;
    int ncor;
    vector <double>  acc;
    void olayer(int nn,int tt,struct oparam &ou_param);
    int setlth(int m){
        lth=m;
        return 0;
    }
    void init_acc(vector <double> &acc,int m);
    int setncor(int m){
        ncor=m;
        return 0;
    }
};
void Olayer::init_acc(vector<double> &acc, int m){
    for(int i=0;i<m;i++){
        acc.push_back(0.0);
    }
}

void Olayer::olayer(int nn,int tt, oparam &ou_param){
    init(ou_param.op,tt,nn);
    vector <double> m(nn,0.0);
    ou_param.label=m;//init label
    init(ou_param.error,tt);
}
void get_label(int m,vector <double> &label){
    for(int i=0;i<8;i++){
        if(m==i){
            label[i]=1.0;
        }
        else{
            label[i]=0.0;
        }
    }
}

void weight_change(vector <vector <double> > &weight,vector <vector <double> > &bsdw,double lr,int m,int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            weight[i][j]=weight[i][j] - lr * bsdw[i][j];
            bsdw[i][j]=0.0;
        }
    }
}
void shuffle(vector <int> &a,vector <int> &b){
    for(int i=0;i<64;i++){
        int randnum=rand()%64;
        int tmp1=a[i];
        int tmp2=b[i];
        a[i]=a[randnum];
        a[randnum]=tmp1;
        b[i]=b[randnum];
        b[randnum]=tmp2;
    }
}
void init_weight(int m,int n,vector <vector <double> > &weight){
    vector <vector <double> > param(m,vector<double>(n,0.0));
    weight=param;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            double a=getrand();
            weight[i][j]=a;
        }
    }
}
void sum_dw(vector <vector <double> > &dweight,int m,int n,vector <vector <double> > &dbsweight){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            dbsweight[i][j]+=dweight[i][j];
            dweight[i][j]=0.0;
        }
    }
}
void sum_db(vector <double> &dba,int m,vector <double> &bsdb){
    for(int i=0;i<m;i++){
        bsdb[i]+=dba[i];
        dba[i]=0.0;
    }
}

void layer_forward(Ilayer &inputl,Hlayer &hiddenl,Olayer &outputl,vector <vector <double> > weight_ih,
                   vector <vector <double> > weight_hh,vector <vector <double> >weight_ho,vector <double> bias,int flag){
    int tt=inputl.tt;
    int inn=inputl.nn;
    int hnn=hiddenl.nn;
    int onn=outputl.nn;
    outputl.setncor(0);
    outputl.setlth(flag);
    for(int i=0;i<tt;i++){
        for(int j=0;j<hnn;j++){
            double wb=0.0;
            hiddenl.hi_param.at[i][j]=0.0;
            for(int k=0;k<inn;k++){
                hiddenl.hi_param.at[i][j]+=inputl.in_param.data[i][k]*weight_ih[j][k]+bias[j];
            }
            if(i==0){
                hiddenl.hi_param.at[i][j]=hiddenl.hi_param.at[i][j]+bias[j];
            }
            else{
                for(int hn=0;hn<hiddenl.nn;hn++){
                    wb=wb+weight_hh[j][hn]*hiddenl.hi_param.bt[i-1][hn];
                }
                hiddenl.hi_param.at[i][j]=hiddenl.hi_param.at[i][j]+bias[j]+wb;
            }

            hiddenl.hi_param.bt[i][j]=sigmoid(hiddenl.hi_param.at[i][j]);
        }

        for(int m=0;m<onn;m++){
            outputl.ou_param.op[i][m]=0.0;
            for(int n=0;n<hnn;n++){
                outputl.ou_param.op[i][m]+=weight_ho[m][n]*hiddenl.hi_param.bt[i][n];
            }
        }
        double max=-9999;
        int maxflag=0;
        for(int ww=0;ww<onn;ww++){
            if(max<outputl.ou_param.op[i][ww]){
                max=outputl.ou_param.op[i][ww];
                maxflag=ww;
            }
        }
        if(maxflag==outputl.lth){
            outputl.ncor+=1;
        }
        double fenmu=0.0;
        for(int kk=0;kk<onn;kk++){
            fenmu=fenmu+exp(outputl.ou_param.op[i][kk]-max);
        }
        for(int p=0;p<onn;p++){
            outputl.ou_param.op[i][p]=(exp(outputl.ou_param.op[i][p]-max))/fenmu;
        }
        double tmp=0.0;
        for(int q=0;q<onn;q++){
            if(outputl.ou_param.label[q]==0){
                tmp+=0.0;
            }
            else{
                tmp=tmp+outputl.ou_param.label[q]*log(outputl.ou_param.label[q]/outputl.ou_param.op[i][q]);
            }
        }
        outputl.ou_param.error[i]=tmp;
    }
    outputl.ou_param.errorall=0.0;
    for(int i1=0;i1<tt;i1++){
        outputl.ou_param.errorall+=outputl.ou_param.error[i1];
    }
}
void layer_backward(Ilayer &inputl,Hlayer &hiddenl,Olayer &outputl,vector <vector <double> > &weight_ih,
                    vector <vector <double> > &weight_hh,vector <vector <double> > &weight_ho){
    int tt=inputl.tt;
    int inn=inputl.nn;
    int hnn=hiddenl.nn;
    int onn=outputl.nn;
    for(int i=tt-1;i>=0;i--){
        for(int j=0;j<hnn;j++){
            hiddenl.hi_param.dht[i][j]=0.0;
            double tmp=0.0;
            for(int k=0;k<onn;k++){
                hiddenl.hi_param.dht[i][j]+=(outputl.ou_param.op[i][k]-outputl.ou_param.label[k])*weight_ho[k][j];
            }
            if(i==tt-1){
                hiddenl.hi_param.dht[i][j]=fsigmoid(hiddenl.hi_param.bt[i][j])*hiddenl.hi_param.dht[i][j];
            }
            else{
                for(int i1=0;i1<hnn;i1++){
                    tmp+=weight_hh[i1][j]*hiddenl.hi_param.dht[i+1][i1];
                }
                hiddenl.hi_param.dht[i][j]=fsigmoid(hiddenl.hi_param.bt[i][j])*(hiddenl.hi_param.dht[i][j]+tmp);
            }
        }
        for(int p=0;p<hnn;p++){
            for(int q=0;q<onn;q++){
                hiddenl.hi_param.dwho[q][p]=(outputl.ou_param.op[i][q]-outputl.ou_param.label[q])*weight_ho[q][p];
                hiddenl.hi_param.dwaho[q][p]+=hiddenl.hi_param.dwho[q][p];
            }
        }
        for(int p1=0;p1<inn;p1++){
            for(int q1=0;q1<hnn;q1++){
                hiddenl.hi_param.dwih[q1][p1]=hiddenl.hi_param.dht[i][q1]*inputl.in_param.data[i][p1];
                hiddenl.hi_param.dwaih[q1][p1]+=hiddenl.hi_param.dwih[q1][p1];
            }
        }
        for(int p2=0;p2<hnn;p2++){
            double tmp3=0.0;
            for(int q2=0;q2<onn;q2++){
                tmp3+=weight_ho[q2][p2]*(outputl.ou_param.op[i][q2]-outputl.ou_param.label[q2]);
            }
            if(i==tt-1){
                hiddenl.hi_param.db[i][p2]=fsigmoid(hiddenl.hi_param.bt[i][p2])*(tmp3);
            }
            else{
                double tmp4=0.0;
                for(int a2=0;a2<hnn;a2++){
                    tmp4+=weight_hh[a2][p2]*hiddenl.hi_param.dht[i+1][a2];
                }
                hiddenl.hi_param.db[i][p2]=fsigmoid(hiddenl.hi_param.bt[i][p2])*(tmp3+tmp4);
            }
        }
    }
    for(int i2=tt-1;i2>=0;i2--){
        for(int j2=0;j2<hnn;j2++){//ben jie dian
            for(int k2=0;k2<hnn;k2++){
                if(i2==0){
                    hiddenl.hi_param.dwahh[j2][k2]+=0.0;
                }
                else{
                    hiddenl.hi_param.dwhh[j2][k2]=hiddenl.hi_param.dht[i2][j2]*hiddenl.hi_param.bt[i2-1][k2];
                    hiddenl.hi_param.dwahh[j2][k2]+=hiddenl.hi_param.dwhh[j2][k2];
                }
            }
            hiddenl.hi_param.dba[j2]+=hiddenl.hi_param.db[i2][j2];
        }
    }
    sum_dw(hiddenl.hi_param.dwaih,hnn,inn,hiddenl.hi_param.bsdih);
    sum_dw(hiddenl.hi_param.dwahh,hnn,hnn,hiddenl.hi_param.bsdhh);
    sum_dw(hiddenl.hi_param.dwaho,onn,hnn,hiddenl.hi_param.bsdho);
    sum_db(hiddenl.hi_param.dba,hnn,hiddenl.hi_param.bsdb);
}

void init_dwall(Ilayer &inputl,Hlayer &hiddenl,Olayer &outputl){
    init_dw(hiddenl.hi_param.dwih,hiddenl.nn,inputl.nn);
    init_dw(hiddenl.hi_param.dwaih,hiddenl.nn,inputl.nn);
    init_dw(hiddenl.hi_param.dwhh,hiddenl.nn,hiddenl.nn);
    init_dw(hiddenl.hi_param.dwahh,hiddenl.nn,hiddenl.nn);
    init_dw(hiddenl.hi_param.dwho,outputl.nn,hiddenl.nn);
    init_dw(hiddenl.hi_param.dwaho,outputl.nn,hiddenl.nn);
    init_dw(hiddenl.hi_param.bsdih,hiddenl.nn,inputl.nn);
    init_dw(hiddenl.hi_param.bsdhh,hiddenl.nn,hiddenl.nn);
    init_dw(hiddenl.hi_param.bsdho,outputl.nn,hiddenl.nn);
}
void init_bias(vector <double> &bias,int m){
    for(int i=0;i<m;i++){
        double a=getrand();
        bias.push_back(a);
    }
}
void update_weight(vector <vector <double> > &weight,vector <vector <double> > &bsweight,int m,int n,double lr,int bs){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            weight[i][j]=weight[i][j]-lr*(bsweight[i][j]/bs);
            bsweight[i][j]=0.0;
        }
    }
}
void update_bias(vector <double> &bias,vector <double> &bsbias,int m,double lr,int bs){
    for(int i=0;i<m;i++){
        bias[i]=bias[i]-lr*(bsbias[i]/bs);
        bsbias[i]=0.0;
    }
}
void display(Olayer outputl){
    double errorall=outputl.ou_param.errorall;
    double error;
    error=errorall/outputl.tt;
    cout<<"error="<<error<<endl;
    ofstream ofile;
    ofile.open("/home/zzq/code/mdono/log",ios::app);
    ofile<<"error="<<error<<endl;
    ofile.close();
}
void test(Ilayer inputl,Hlayer hiddenl,Olayer outputl,int epoch,vector <vector <double> > weight_ih,
          vector <vector <double> >weight_hh,vector <vector <double> > weight_ho,vector <double> bias,double *ptr,int *p){
    for(int i=0;i<outputl.nn;i++){
        outputl.acc[i]=0.0;
    }
    for(int i=0;i<epoch;i++){
        int label=*p;
        p++;
        outputl.setlth(label);
        get_label(label,outputl.ou_param.label);
        for(int j=0;j<inputl.tt;j++){
            for(int k=0;k<inputl.nn;k++){
                inputl.in_param.data[j][k]=*ptr;
                ptr++;
            }
        }
        layer_forward(inputl,hiddenl,outputl,weight_ih,weight_hh,weight_ho,bias,label);
        int lth=outputl.lth;
        int ncor=outputl.ncor;
        outputl.acc[lth]=outputl.acc[lth]+(double)ncor/outputl.tt;
    }
    for(int i=0;i<8;i++){
        outputl.acc[i]=outputl.acc[i]/8;
    }
    double accall=0.0;
    for(int i=0;i<8;i++){
        accall+=outputl.acc[i];
    }
    accall=accall/8;
    cout<<"acc="<<accall<<endl;
    ofstream ofile;
    ofile.open("/home/zzq/code/mdono/log",ios::app);
    ofile<<"acc="<<accall<<endl;
    ofile.close();
}
void write_weight(vector <vector <double> > weight,int m,int n){
    ofstream ofile;
    ofile.open("param",ios::app);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            ofile<<weight[i][j]<<" ";
        }
        ofile<<endl;
    }
    ofile.close();
}
void write_bias(vector <double> bias,int m){
    ofstream ofile;
    ofile.open("param",ios::app);
    for(int i=0;i<m;i++){
        ofile<<bias[i]<<" ";
    }
    ofile<<endl;
}

void write_param(Ilayer inputl,Hlayer hiddenl,Olayer outputl,vector <vector <double> >weight_ih
                 ,vector <vector <double> > weight_hh,vector <vector <double> > weight_ho,vector <double> bias){
    write_weight(weight_ih,hiddenl.nn,inputl.nn);
    write_weight(weight_hh,hiddenl.nn,hiddenl.nn);
    write_weight(weight_ho,outputl.nn,hiddenl.nn);
    write_bias(bias,hiddenl.nn);
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    int iter=5000;
    double lr=0.001;
    int bs=8;
    int epoch=64;
    int tt=300;
    int test_iter=10;
    double data[64][300][6];
    ifstream ifile;
    ifile.open("/home/zzq/code/mdono/data");
    for(int i=0;i<epoch;i++){
        for(int j=0;j<300;j++){
            for(int k=0;k<6;k++){
                ifile>>data[i][j][k];
            }
        }
    }
    ifile.close();
    vector <int> c(64,0);
    for(int i=0;i<epoch;i++){
        c[i]=i;
    }
    ifstream lfile;
    lfile.open("/home/zzq/code/mdono/label");
    vector <int>  label(64,0);
    for(int i=0;i<epoch;i++){
        lfile>>label[i];
    }
    lfile.close();
    ifstream pfile;
    pfile.open("/home/zzq/code/mdono/label");
    int shun[64];
    for(int i=0;i<epoch;i++){
        pfile>>shun[i];
    }
    pfile.close();
    shuffle(c,label);
    Ilayer inputl;
    Hlayer hiddenl;
    Olayer outputl;
    int inum=6;
    int hnum=300;
    int onum=8;
    inputl.set_nn(6);
    hiddenl.set_nn(300);
    outputl.set_nn(8);
    hiddenl.set_bs(8);
    inputl.set_tt(tt);
    hiddenl.set_tt(tt);
    outputl.set_tt(tt);
    inputl.ilayer(inputl.nn,inputl.tt,inputl.in_param);
    hiddenl.hlayer(hiddenl.nn,hiddenl.tt,hiddenl.hi_param);
    outputl.olayer(outputl.nn,outputl.tt,outputl.ou_param);
    vector <vector <double> > weight_ih;
    vector <vector <double> > weight_hh;
    vector <vector <double> > weight_ho;
    init_weight(hnum,inum,weight_ih);
    init_weight(hnum,hnum,weight_hh);
    init_weight(onum,hnum,weight_ho);
    init_dwall(inputl,hiddenl,outputl);
    outputl.init_acc(outputl.acc,outputl.nn);//acc leiwai
    vector <double> bias;
    init_bias(bias,hiddenl.nn);
    for(int i=0;i<iter;i++){
        for(int j=0;j<epoch;j++){
            int id=c[j];//from 0 to 63
            double *ptr=data[id][0];
            for(int mi=0;mi<tt;mi++){
                for(int mj=0;mj<inum;mj++){
                    inputl.in_param.data[mi][mj]=*ptr;
                    ptr++;
                }
            }
            get_label(shun[id],outputl.ou_param.label);//she zhi label
            layer_forward(inputl,hiddenl,outputl,weight_ih,weight_hh,weight_ho,bias,shun[id]);
            layer_backward(inputl,hiddenl,outputl,weight_ih,weight_hh,weight_ho);
            hiddenl.countjia();
            if(hiddenl.count==bs){
               update_weight(weight_ih,hiddenl.hi_param.bsdih,hiddenl.nn,inputl.nn,lr,bs);
               update_weight(weight_hh,hiddenl.hi_param.bsdhh,hiddenl.nn,hiddenl.nn,lr,bs);
               update_weight(weight_ho,hiddenl.hi_param.bsdho,outputl.nn,hiddenl.nn,lr,bs);
               update_bias(bias,hiddenl.hi_param.bsdb,hiddenl.nn,lr,bs);
               hiddenl.zero_count();
               display(outputl);
            }
        }
        if(i%test_iter==0){
            double *ptr1=data[0][0];
            int *ptr2=shun;
            test(inputl,hiddenl,outputl,epoch,weight_ih,weight_hh,weight_ho,bias,ptr1,ptr2);
        }
    }
    write_param(inputl,hiddenl,outputl,weight_ih,weight_hh,weight_ho,bias);
    return a.exec();
}
