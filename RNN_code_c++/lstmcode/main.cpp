#include <QCoreApplication>
#include<iostream>
#include<fstream>
#include<math.h>
#include<time.h>
#include<vector>
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
inline double tanh(double x){
    return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}
inline double ftanh(double x){
    return 1-x*x;
}
void get_rng(vector <double> &param,int m){
    for(int i=0;i<m;i++){
        double a=getrand();
        param.push_back(a);
    }
}
void init_weight(vector <vector <double> > &weight,int m,int n){
    vector <vector <double> > param(m,vector<double>(n,0.0));
    weight=param;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            double a=getrand();
            weight[i][j]=a;
        }
    }
}


class layer{
public:
    int t;
    int nn;
    int samth;
};
class Ilayer:public layer{
public:
    int ns;
    vector <vector <double> > weight_ih;
    vector <vector <vector <double> > > data;
    Ilayer(int k,int m,int n){
        t=m;
        nn=n;
        ns=k;
        vector <vector <vector <double> > > v(ns,vector <vector <double> >(t,vector <double>(nn,0)));
        data=v;
        ifstream ifile;
        ifile.open("data");
        for(int i=0;i<ns;i++){
            for(int j=0;j<t;j++){
                for(int k=0;k<nn;k++){
                    ifile>>data[i][j][k];
                }
            }
        }
    }
};
class Hlayer:public layer{
public:
    vector <vector <double> > weight_hh;
    vector <vector <double> > weight_ho;
    vector <vector <double> > aht;
    vector <vector <double> > bht;
    vector <double> hb;
    Hlayer(int m,int n){
        t=m;
        nn=n;
        vector <vector <double> > param(t,vector<double>(nn,0));
        aht=param;
        bht=param;
        get_rng(hb,nn);
    }
};
class Olayer:public layer{
public:
    int ns;
    vector <vector <double> > akt;
    vector <double> yflag;
    vector <vector <double> > reror;
    vector <double> error;
    double errorall;
    vector <vector <vector <double> > > label;
    Olayer(int k,int m,int n){
        ns=k;
        t=m;
        nn=n;
        vector <vector <vector <double> > > param(ns,vector <vector <double> >(t,vector<double>(nn,0)));
        label=param;
        ifstream ifile;
        ifile.open("label");
        for(int i=0;i<ns;i++){
            for(int j=0;j<t;j++){
                for(int k=0;k<nn;k++){
                    ifile>>label[i][j][k];
                }
            }
        }
        vector <vector <double> > vv(t,vector<double>(nn,0));
        akt=vv;
        reror=vv;
        errorall=0.0;
        for(int i=0;i<t;i++){
            error.push_back(0.0);
        }
        vector <double> param2(nn,0.0);
        yflag=param2;
    }
    void init_yflag(vector <double> &yflag,int m,int n);
};
void Olayer::init_yflag(vector <double> &yflag,int m,int n){
    for(int i=0;i<m;i++){
        if(n==i){
            yflag[i]=1.0;
        }
        else{
            yflag[i]=0;
        }
    }
}

class Block{
public:
    class igate{
    public:
        vector <double> wil;
        vector <double> whl;
        double wcl;
        vector <double> alt;
        vector <double> blt;
        double ib;
        igate(int inum,int hnum,int t){
            get_rng(wil,inum);
            get_rng(whl,hnum);
            vector <double> param(t,0.0);
            alt=param;
            blt=param;
            wcl=getrand();
            ib=getrand();
        }
    };
    class fgate{
    public:
        vector <double> wif;
        vector <double> whf;
        double wcf;
        vector <double> aft;
        vector <double> bft;
        double fb;
        fgate(int inum,int hnum,int t){
            get_rng(wif,inum);
            get_rng(whf,hnum);
            vector <double> param(t,0.0);
            aft=param;
            bft=param;
            wcf=getrand();
            fb=getrand();
        }
    };
    class wgate{
    public:
        vector <double> wiw;
        vector <double> whw;
        double wcw;
        vector <double> awt;
        vector <double> bwt;
        double wb;
        wgate(int inum,int hnum,int t){
            get_rng(wiw,inum);
            get_rng(whw,hnum);
            vector <double> param(t,0.0);
            awt=param;
            bwt=param;
            wcw=getrand();
            wb=getrand();
        }
    };
    class cell{
     public:
        vector <double> wic;
        vector <double> whc;
        vector <double> act;
        vector <double> sct;
        vector <double> bct;
        double cb;
        cell(int inum,int hnum,int t){
            get_rng(wic,inum);
            get_rng(whc,hnum);
            vector <double> param(t,0.0);
            act=param;
            sct=param;
            bct=param;
            cb=getrand();
        }
    };
    int nb;
    vector <igate> ig;
    vector <fgate> fg;
    vector <wgate> wg;
    vector <cell> cl;
    vector <vector <double> > weight_co;
    Block(int m,int inum,int hnum,int onum,int t){
        nb=m;
        init_weight(weight_co,onum,hnum);
        igate input_gate(inum,hnum,t);
        fgate forget_gate(inum,hnum,t);
        wgate output_gate(inum,hnum,t);
        cell cell_param(inum,hnum,t);
        for(int i=0;i<nb;i++){
            ig.push_back(input_gate);
            fg.push_back(forget_gate);
            wg.push_back(output_gate);
            cl.push_back(cell_param);
        }
    }
};
void layer_weight(Ilayer &inputl,Hlayer &hiddenl,Olayer &outputl){
    init_weight(inputl.weight_ih,hiddenl.nn,inputl.nn);
    init_weight(hiddenl.weight_hh,hiddenl.nn,hiddenl.nn);
    init_weight(hiddenl.weight_ho,outputl.nn,hiddenl.nn);
}
void layer_forward(Ilayer &inputl,Hlayer &hiddenl,Olayer &outputl,Block &block){
    outputl.init_yflag(outputl.yflag,outputl.nn,outputl.samth);
    int th=inputl.samth;
    int inn=inputl.nn;
    int hnn=hiddenl.nn;
    int onn=outputl.nn;
    int nb=block.nb;
    for(int i=0;i<inputl.t;i++){
        for(int j=0;j<hiddenl.nn;j++){
            double wb=0.0;
            hiddenl.aht[i][j]=0.0;
            for(int k=0;k<inn;k++){
                hiddenl.aht[i][j]+=inputl.data[th][i][k]*inputl.weight_ih[j][k];
            }
            if(i==0){
                hiddenl.bht[i][j]=sigmoid(hiddenl.aht[i][j]+hiddenl.hb[j]);
            }
            else{
                for(int l=0;l<hnn;l++){
                    wb+=hiddenl.weight_hh[j][l]*hiddenl.bht[i-1][l];
                }
                hiddenl.bht[i][j]=sigmoid(wb+hiddenl.aht[i][j]+hiddenl.hb[j]);
            }
        }
        for(int a1=0;a1<nb;a1++){
            block.ig[a1].alt[i]=0.0;
            block.fg[a1].aft[i]=0.0;
            block.cl[a1].act[i]=0.0;
            block.cl[a1].sct[i]=0.0;
            block.wg[a1].awt[i]=0.0;
            block.cl[a1].bct[i]=0.0;
            for(int b1=0;b1<inn;b1++){
                block.ig[a1].alt[i]+=block.ig[a1].wil[b1]*inputl.data[th][i][b1];
                block.fg[a1].aft[i]+=block.fg[a1].wif[b1]*inputl.data[th][i][b1];
                block.cl[a1].act[i]+=block.cl[a1].wic[b1]*inputl.data[th][i][b1];
                block.cl[a1].awt[i]+=block.wg[a1].wiw[b1]*inputl.data[th][i][b1];
            }
            if(i==0){
                block.ig[a1].alt[i]+=block.ig[a1].ib;
                block.fg[a1].aft[i]+=block.fg[a1].fb;
                block.cl[a1].act[i]+=block.cl[a1].cb;
            }
            else{
                double tmpl;
                double tmpf;
                double tmpc;
                double tmpw;
                for(int c1=0;c1<hnn;c1++){
                   tmpl+=block.ig[a1].whl[c1]*hiddenl.bht[i-1][c1];
                   tmpf+=block.fg[a1].whf[c1]*hiddenl.bht[i-1][c1];
                   tmpc+=block.cl[a1].whc[c1]*hiddenl.bht[i-1][c1];
                   tmpw+=block.wg[a1].whw[c1]*hiddenl.bht[i-1][c1];
                }
                double tmpl1;
                double tmpf1;
                double tmpw1;
                tmpl1=block.ig[a1].wcl*block.cl[a1].sct[i-1];
                tmpf1=block.fg[a1].wcf*block.cl[a1].sct[i-1];
                tmpw1=block.wg[a1].wcw*block.cl[a1].sct[i-1];
                block.ig[a1].alt[i]=block.ig[a1].alt[i]+tmpl+tmpl1+block.ig[a1].ib;
                block.fg[a1].aft[i]=tmpf+tmpf1+block.fg[a1].aft[i]+block.fg[a1].fb;
                block.cl[a1].act[i]=block.cl[a1].act[i]+tmpc+block.cl[a1].cb;
                block.wg[a1].awt[i]=block.wg[a1].awt[i]+tmpw+tmpw1+block.wg[a1].wb;
            }
            block.ig[a1].blt[i]=sigmoid(block.ig[a1].alt[i]);
            block.fg[a1].bft[i]=sigmoid(block.fg[a1].aft[i]);
            block.wg[a1].bwt[i]=sigmoid(block.wg[a1].awt[i]);
            if(i==0){
                block.cl[a1].sct[i]=block.cl[a1].sct[i]+block.ig[a1].blt[i]*tanh(block.cl[a1].act[i]);
            }
            else{
                block.cl[a1].sct[i]=block.cl[a1].sct[i]+block.ig[a1].blt[i]*tanh(block.cl[a1].act[i])
                        +block.fg[a1].bft[i]*block.cl[a1].sct[i-1];
            }
            block.cl[a1].bct[i]=block.wg[a1].bwt[i]*tanh(block.cl[a1].sct[i]);
        }
        for(int a2=0;a2<onn;a2++){
            outputl.akt[i][a2]=0.0;
            for(int b2=0;b2<nb;b2++){
                outputl.akt[i][a2]+=block.weight_co[a2][b2]*block.cl[b2].bct[i];
            }
            outputl.reror[i][a2]=outputl.akt[i][a2]-outputl.yflag[a2];
        }
        outputl.error[i]=0.0;
        for(int c2=0;c2<onn;c2++){
            outputl.error[i]+=outputl.reror[i][c2];
        }
    }
    outputl.errorall=0.0;
    for(int i1=0;i1<t;i1++){
        outputl.errorall+=outputl.error[i1];
    }
}
void layer_backward(Ilayer &inputl,Hlayer &hiddenl,Olayer &outputl,Block &block){
    int inn=inputl.nn;
    int hnn=hiddenl.nn;
    int onn=outputl.nn;
    int t=inputl.t;
    for(int i=t-1;t>=0;t--){

        for(int j=0;j<onn;j++){

        }
    }
}
void write_weight(Ilayer inputl,Hlayer hiddenl,Olayer outputl){
    ofstream ofile;
    ofile.open("param",ios::app);

}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    int epoch=1000;
    int ns=64;
    double lr=0.01;
    int batch=10;
    int inum=6;
    int hnum=300;
    int onum=8;
    int nb=100;
    int t=300;
    Ilayer inputl(ns,t,inum);
    Hlayer hiddenl(t,hnum);
    Olayer outputl(ns,t,onum);
    Block block(nb,inum,hnum,onum,t);
    layer_weight(inputl,hiddenl,outputl);
    for(int i=0;i<ns;i++){
        outputl.samth=i;
        inputl.samth=i;
        layer_forward(inputl,hiddenl,outputl,block);
    }
    return a.exec();
}
