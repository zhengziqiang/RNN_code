#include <QCoreApplication>
#include<iostream>
#include<fstream>
#include<math.h>
#include<time.h>
#include<vector>
#include<string>
using namespace std;
struct dparam{
    vector <vector <double> > ect;
    vector <vector <double> > est;
    vector <vector <double> > dwt;
    vector <vector <double> > dct;
    vector <vector <double> > dft;
    vector <vector <double> > dlt;
};
void sum_db(double &db,double &bsdb){
    bsdb+=db;
   db=0;
}

void sum_dweightone(vector <double> &dweight,vector <double> &bsdweight,int m){
    for(int i=0;i<m;i++){
        bsdweight[i]+=dweight[i];
        dweight[i]=0;
    }
}
void sum_dweight(vector <vector <double> > &dweight,vector <vector <double> > &bsdweight,int onn,int nb){
    for(int i=0;i<onn;i++){
        for(int j=0;j<nb;j++){
            bsdweight[i][j]+=dweight[i][j];
            dweight[i][j]=0;
        }
    }
}

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

//sigmoid函数
inline double sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}
//sigmoid的导函数
inline double fsigmoid(double x){
    return x*(1-x);
}
inline double tanh(double x){
    return ((exp(x)-exp(-x))/(exp(x)+exp(-x)));
}
inline double ftanh(double x){
    return 1-((exp(x)-exp(-x))/(exp(x)+exp(-x)))*((exp(x)-exp(-x))/(exp(x)+exp(-x)));
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
    vector <vector <vector <double> > > data;
    Ilayer(int k,int m,int n){
        t=m;
        nn=n;
        ns=k;
        vector <vector <vector <double> > > v(ns,vector <vector <double> >(t,vector <double>(nn,0)));
        data=v;
        ifstream ifile;
        ifile.open("/home/zzq/code/lstmdebug/data");
        for(int i=0;i<ns;i++){
            for(int j=0;j<t;j++){
                for(int k=0;k<nn;k++){
                    ifile>>data[i][j][k];
                }
            }
        }
    }
};

class Olayer:public layer{
public:
    int ns;
    int ncor;
    int lth;
    int count;
    vector <double> acc;
    vector <vector <double> > akt;
    vector <double> yflag;
    vector <vector <double> > reror;
    vector <double> error;
    double errorall;
    vector <int> label;
    int set_lth(int m){
        lth=m;
        return 0;
    }
    int set_ncor(int m){
        ncor=m;
        return 0;
    }
    int countjia(){
        count+=1;
        return 0;
    }
    int zero_count(){
        count=0;
        return 0;
    }

    Olayer(int k,int m,int n){
        ns=k;
        t=m;
        nn=n;
        vector <double> param(ns,0);
        vector <int> paramlabel(ns,0);
        label=paramlabel;
        ifstream ifile;
        ifile.open("/home/zzq/code/lstmdebug/label");
        for(int i=0;i<ns;i++){
            ifile>>label[i];
        }
        ifile.close();
        vector <double> ac(8,0);
        acc=ac;
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
        vector <double> wcl;
        vector <double> alt;
        vector <double> blt;
        vector <double> dwil;
        vector <double> dwhl;
        vector <double> dwcl;
        vector <double> bsdwil;
        vector <double> bsdwhl;
        vector <double> bsdwcl;
        double dib;
        double ib;
        double bsdib;
        igate(int inum,int hnum,int t){
            get_rng(wil,inum);
            get_rng(whl,hnum);
            get_rng(wcl,hnum);
            vector <double> param(t,0.0);
            vector <double> param_inn(inum,0);
            vector <double> param_hnn(hnum,0);
            alt=param;
            blt=param;
            ib=getrand();
            dib=0;
            bsdib=0;
            dwil=param_inn;
            dwhl=param_hnn;
            dwcl=param_hnn;
            bsdwil=param_inn;
            bsdwhl=param_hnn;
            bsdwcl=param_hnn;
        }
    };
    class fgate{
    public:
        vector <double> wif;
        vector <double> whf;
        vector <double> wcf;
        vector <double> dwif;
        vector <double> dwhf;
        vector <double> dwcf;
        vector <double> aft;
        vector <double> bft;
        vector <double> bsdwif;
        vector <double> bsdwhf;
        vector <double> bsdwcf;
        double fb;
        double dfb;
        double bsdfb;
        fgate(int inum,int hnum,int t){
            get_rng(wif,inum);
            get_rng(whf,hnum);
            vector <double> param(t,0.0);
            vector <double> param_inn(inum,0);
            vector <double> param_hnn(hnum,0);
            aft=param;
            bft=param;
            get_rng(wcf,hnum);
            fb=getrand();
            dfb=0;
            bsdfb=0;
            dwif=param_inn;
            dwhf=param_hnn;
            dwcf=param_hnn;
            bsdwif=param_inn;
            bsdwhf=param_hnn;
            bsdwcf=param_hnn;
        }
    };
    class wgate{
    public:
        vector <double> wiw;
        vector <double> whw;
        vector <double> wcw;
        vector <double> dwiw;
        vector <double> dwhw;
        vector <double> dwcw;
        vector <double> bsdwiw;
        vector <double> bsdwhw;
        vector <double> bsdwcw;
        vector <double> awt;
        vector <double> bwt;
        double wb;
        double dwb;
        double bsdwb;
        wgate(int inum,int hnum,int t){
            get_rng(wiw,inum);
            get_rng(whw,hnum);
            vector <double> param(t,0.0);
            vector <double> param_hnn(hnum,0.0);
            vector <double> param_inn(inum,0);
            awt=param;
            bwt=param;
            get_rng(wcw,hnum);
            wb=getrand();
            dwb=0;
            bsdwb=0;
            dwiw=param_inn;
            dwhw=param_hnn;
            dwcw=param_hnn;
            bsdwiw=param_inn;
            bsdwhw=param_hnn;
            bsdwcw=param_hnn;
        }
    };
    class cell{
     public:
        vector <double> wic;
        vector <double> whc;
        vector <double> dwic;
        vector <double> dwhc;
        vector <double> bsdwic;
        vector <double> bsdwhc;
        vector <double> act;
        vector <double> sct;
        vector <double> bct;
        double cb;
        double dcb;
        double bsdcb;
        cell(int inum,int hnum,int t){
            get_rng(wic,inum);
            get_rng(whc,hnum);
            vector <double> param(t,0.0);
            vector <double> param_hnn(hnum,0.0);
            vector <double> param_inn(inum,0);
            act=param;
            sct=param;
            bct=param;
            dwic=param_inn;
            dwhc=param_hnn;
            cb=getrand();
            dcb=0;
            bsdcb=0;
            bsdwic=param_inn;
            bsdwhc=param_hnn;
        }
    };
    int nb;
    vector <igate> ig;
    vector <fgate> fg;
    vector <wgate> wg;
    vector <cell> cl;
    struct dparam dam;
    vector <vector <double> > weight_co;
    vector <vector <double> > dwco;
    vector <vector <double> > bsdwco;
    void init_dam(struct dparam &dam,int t,int m);
    Block(int m,int inum,int onum,int t){
        nb=m;
        init_weight(weight_co,onum,nb);
        vector <vector <double> > param(onum,vector <double>(nb,0.0));
        vector <vector <double> > param2(t,vector <double>(nb,0));
        dam.ect=param2;
        dam.est=param2;
        dam.dwt=param2;
        dam.dct=param2;
        dam.dft=param2;
        dam.dlt=param2;
        dwco=param;
        bsdwco=param;
        for(int i=0;i<nb;i++){
            igate input_gate(inum,nb,t);
            fgate forget_gate(inum,nb,t);
            wgate output_gate(inum,nb,t);
            cell cell_param(inum,nb,t);
            ig.push_back(input_gate);
            fg.push_back(forget_gate);
            wg.push_back(output_gate);
            cl.push_back(cell_param);
        }
    }
};
void Block::init_dam(dparam &dam,int t,int m){
    vector <vector <double> > param(t,vector <double>(m,0));
    dam.dct=param;
    dam.dft=param;
    dam.dlt=param;
    dam.dwt=param;
    dam.ect=param;
    dam.est=param;
}

void layer_forward(Ilayer &inputl,Olayer &outputl,Block &block){
    int lth=outputl.label[inputl.samth];
    outputl.set_lth(lth);
    outputl.init_yflag(outputl.yflag,outputl.nn,lth);
    int th=inputl.samth;
    int inn=inputl.nn;
    int onn=outputl.nn;
    int nb=block.nb;
    int t=inputl.t;
    outputl.set_ncor(0);
    for(int i=0;i<t;i++){
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
                block.wg[a1].awt[i]+=block.wg[a1].wiw[b1]*inputl.data[th][i][b1];
            }

            if(i==0){
                block.ig[a1].alt[i]+=block.ig[a1].ib;
                block.fg[a1].aft[i]+=block.fg[a1].fb;
                block.cl[a1].act[i]+=block.cl[a1].cb;
            }
            else{
                double tmpl=0;
                double tmpf=0;
                double tmpc=0;
                double tmpw=0;
                double tmpl1=0;
                double tmpf1=0;
                double tmpw1=0;
                for(int c1=0;c1<nb;c1++){
                   tmpl+=block.ig[a1].whl[c1]*block.cl[a1].bct[i-1];
                   tmpf+=block.fg[a1].whf[c1]*block.cl[a1].bct[i-1];
                   tmpc+=block.cl[a1].whc[c1]*block.cl[a1].bct[i-1];
                   tmpw+=block.wg[a1].whw[c1]*block.cl[a1].bct[i-1];
                   //
                   //
                }
                tmpl1+=block.ig[a1].wcl[a1]*block.cl[a1].sct[i-1];
                tmpf1+=block.fg[a1].wcf[a1]*block.cl[a1].sct[i-1];
                tmpw1+=block.wg[a1].wcw[a1]*block.cl[a1].sct[i-1];

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
        }
        double max=-9999;
        int maxflag=0;
        for(int ww=0;ww<onn;ww++){
            if(max<outputl.akt[i][ww]){
                max=outputl.akt[i][ww];
                maxflag=ww;
            }
        }
        if(maxflag==outputl.lth){
            outputl.ncor+=1;
        }
        double fenmu=0;
        for(int kk=0;kk<onn;kk++){
            fenmu=fenmu+exp(outputl.akt[i][kk]-max);
        }
        for(int p=0;p<onn;p++){
            outputl.akt[i][p]=(exp(outputl.akt[i][p]-max))/fenmu;
        }
        double tmperror=0.0;
        for(int q=0;q<onn;q++){
            if(outputl.yflag[q]==0){
                tmperror+=0.0;
            }
            else{
                tmperror=tmperror+outputl.yflag[q]*log(outputl.yflag[q]/outputl.akt[i][q]);
            }
            outputl.reror[i][q]=outputl.akt[i][q]-outputl.yflag[q];
        }
        outputl.error[i]=tmperror;
    }
    outputl.errorall=0.0;
    for(int i1=0;i1<t;i1++){
        outputl.errorall+=outputl.error[i1];
    }
}
void layer_backward(Ilayer &inputl,Olayer &outputl,Block &block){
    int inn=inputl.nn;
    int nb=block.nb;
    int onn=outputl.nn;
    int t=inputl.t;
    int th=outputl.samth;
    for(int i=t-1;i>=0;i--){
        for(int a1=0;a1<nb;a1++){
            block.dam.ect[i][a1]=0;
            for(int b1=0;b1<onn;b1++){
                block.dam.ect[i][a1]+=block.weight_co[b1][a1]*outputl.reror[i][b1];//diyi bufen yijing xiehao
            }
            if(i==t-1){
                block.dam.ect[i][a1]+=0;
            }
            else{
                for(int h1=0;h1<nb;h1++){
                    block.dam.ect[i][a1]+=block.cl[a1].whc[h1]*block.dam.dct[i+1][h1];//
                }
            }

        }//ect bufen  zhengchang

        for(int a2=0;a2<nb;a2++){
            block.dam.dwt[i][a2]=0;
            block.dam.dwt[i][a2]=fsigmoid(block.wg[a2].bwt[i])*tanh(block.cl[a2].sct[i])*block.dam.ect[i][a2];
        }//dwt you sige 0

        for(int a3=0;a3<nb;a3++){
            block.dam.est[i][a3]=0;
            double tmpest=0;
            tmpest=block.wg[a3].bwt[i]*ftanh(block.cl[a3].sct[i])*block.dam.ect[i][a3];
            double tmpbe=0;
            if(i==t-1){
                tmpbe+=0;
            }
            else{
                tmpbe+=block.fg[a3].bft[i+1]*block.dam.est[i+1][a3];
            }
            double tmpwd=0;
            if(i==t-1){
                tmpwd+=0;
            }
            else{
                tmpwd+=block.ig[a3].wcl[a3]*block.dam.dlt[i+1][a3];
            }
            double tmpwcf=0;
            if(i==t-1){
                tmpwcf+=0;
            }
            else{
                tmpwcf+=block.fg[a3].wcf[a3]*block.dam.dft[i+1][a3];
            }
            double tmpwcw=0;
            tmpwcw+=block.wg[a3].wcw[a3]*block.dam.dwt[i][a3];
            block.dam.est[i][a3]=block.dam.est[i][a3]+tmpest+tmpbe+tmpwd+tmpwcf+tmpwcw;
        }//est zhengchang

        for(int a4=0;a4<nb;a4++){
            block.dam.dct[i][a4]=0;
            block.dam.dct[i][a4]=block.ig[a4].blt[i]*ftanh(block.cl[a4].act[i])*block.dam.est[i][a4];
        }//dct zhengchang

        for(int a5=0;a5<nb;a5++){
            double tmpcc=0;
            block.dam.dft[i][a5]=0;
            if(i==0){
                tmpcc+=0;
            }
            else{
                tmpcc+=block.cl[a5].sct[i-1]*block.dam.est[i][a5];
            }
            block.dam.dft[i][a5]=fsigmoid(block.fg[a5].bft[i])*tmpcc;
        }//dft  qizhi

        for(int a6=0;a6<nb;a6++){
            double tmplt=0;
            block.dam.dlt[i][a6]=0;
            tmplt+=tanh(block.cl[a6].act[i])*block.dam.est[i][a6];
            block.dam.dlt[i][a6]=fsigmoid(block.ig[a6].blt[i])*tmplt;
        }//dlt
        //weight delta

        for(int j=0;j<nb;j++){
            for(int k=0;k<onn;k++){
                block.dwco[k][j]=block.dwco[k][j]+outputl.reror[i][k]*block.weight_co[k][j];
            }
        }

        for(int i2=0;i2<nb;i2++){
            for(int j2=0;j2<inn;j2++){
                block.ig[i2].dwil[j2]=block.ig[i2].dwil[j2]+block.dam.dlt[i][i2]*inputl.data[th][i][j2];
                block.fg[i2].dwif[j2]+=block.dam.dft[i][i2]*(inputl.data[th][i][j2]);
                block.cl[i2].dwic[j2]+=block.dam.dct[i][i2]*inputl.data[th][i][j2];
                block.wg[i2].dwiw[j2]+=block.dam.dwt[i][i2]*inputl.data[th][i][j2];
            }
            for(int k2=0;k2<nb;k2++){
                if(i==0){
                    block.ig[i2].dwhl[k2]+=0;
                    block.fg[i2].dwhf[k2]+=0;
                    block.cl[i2].dwhc[k2]+=0;
                    block.wg[i2].dwhw[k2]+=0;
                }
                else{
                    block.ig[i2].dwhl[k2]+=block.cl[i2].bct[i-1]*block.dam.dlt[i][k2];
                    block.fg[i2].dwhf[k2]+=block.cl[i2].bct[i-1]*block.dam.dft[i][k2];
                    block.cl[i2].dwhc[k2]+=block.cl[i2].bct[i-1]*block.dam.dct[i][k2];
                    block.wg[i2].dwhw[k2]+=block.cl[i2].bct[i-1]*block.dam.dwt[i][k2];
                }
            }

            for(int l2=0;l2<nb;l2++){
                if(i==0){
                    block.ig[i2].dwcl[l2]+=0;
                    block.fg[i2].dwcf[l2]+=0;
                }
                else{
                    block.ig[i2].dwcl[l2]+=block.cl[i2].sct[i-1]*block.dam.dlt[i][l2];
                    block.fg[i2].dwcf[l2]+=block.cl[i2].sct[i-1]*block.dam.dft[i][l2];
                }
                block.wg[i2].dwcw[l2]+=block.cl[i2].sct[i]*block.dam.dwt[i][l2];
            }
        }

        for(int b8=0;b8<nb;b8++){
            block.ig[b8].dib+=block.dam.dlt[i][b8]*fsigmoid(block.ig[b8].blt[i]);
            block.fg[b8].dfb+=block.dam.dft[i][b8]*fsigmoid(block.fg[b8].bft[i]);
            block.cl[b8].dcb+=block.dam.dct[i][b8]*ftanh(block.cl[b8].act[i]);
            block.wg[b8].dwb+=block.dam.dwt[i][b8]*fsigmoid(block.wg[b8].bwt[i]);
        }
    }
    sum_dweight(block.dwco,block.bsdwco,onn,nb);
    for(int i3=0;i3<nb;i3++){
        sum_dweightone(block.ig[i3].dwil,block.ig[i3].bsdwil,inn);
        sum_dweightone(block.ig[i3].dwhl,block.ig[i3].bsdwhl,nb);
        sum_dweightone(block.ig[i3].dwcl,block.ig[i3].bsdwcl,nb);
        sum_dweightone(block.fg[i3].dwif,block.fg[i3].bsdwif,inn);
        sum_dweightone(block.fg[i3].dwhf,block.fg[i3].bsdwhf,nb);
        sum_dweightone(block.fg[i3].dwcf,block.fg[i3].bsdwcf,nb);
        sum_dweightone(block.cl[i3].dwic,block.cl[i3].bsdwic,nb);
        sum_dweightone(block.cl[i3].dwhc,block.cl[i3].bsdwhc,nb);
        sum_dweightone(block.wg[i3].dwiw,block.wg[i3].bsdwiw,inn);
        sum_dweightone(block.wg[i3].dwhw,block.wg[i3].bsdwhw,nb);
        sum_dweightone(block.wg[i3].dwcw,block.wg[i3].bsdwcw,nb);
        sum_db(block.ig[i3].dib,block.ig[i3].bsdib);
        sum_db(block.fg[i3].dfb,block.fg[i3].bsdfb);
        sum_db(block.cl[i3].dcb,block.cl[i3].bsdcb);
        sum_db(block.wg[i3].dwb,block.wg[i3].bsdwb);
    }
}
void display(Olayer &outputl){
    double errorall=outputl.errorall;
    double error=0;
    error=errorall/outputl.t;
    cout<<"error="<<error<<endl;
    ofstream ofile;
    ofile.open("/home/zzq/code/lstmdebug/log",ios::app);
    ofile<<"error="<<error<<endl;
    ofile.close();
}
void update_weight(vector <double> &weight,vector <double> &dweight,int m,double lr,int bat){
    for(int i=0;i<m;i++){
        weight[i]= weight[i]-dweight[i]*lr/bat;
        dweight[i]=0;
    }
}
void update_weight(vector <vector <double> > &weight,vector <vector <double> > &dweight,int onn,int nb,double lr,int bat){
    for(int i=0;i<onn;i++){
        for(int j=0;j<nb;j++){
            weight[i][j]=weight[i][j]-dweight[i][j]*lr/bat;
            dweight[i][j]=0;
        }
    }
}
void update_bias(double &bias,double &dbias,double lr,int bat){
    bias=bias-lr*dbias/bat;
    dbias=0;
}

void update(Ilayer &inputl,Olayer &outputl,Block &block,double lr,int bat){
    int nb=block.nb;
    int inn=inputl.nn;
    int onn=outputl.nn;
    for(int i=0;i<nb;i++){
        update_weight(block.ig[i].wil,block.ig[i].bsdwil,inn,lr,bat);
        update_weight(block.ig[i].whl,block.ig[i].bsdwhl,nb,lr,bat);
        update_weight(block.ig[i].wcl,block.ig[i].bsdwcl,nb,lr,bat);
        update_bias(block.ig[i].ib,block.ig[i].bsdib,lr,bat);

        update_weight(block.fg[i].wif,block.fg[i].bsdwif,inn,lr,bat);
        update_weight(block.fg[i].whf,block.fg[i].bsdwhf,nb,lr,bat);
        update_weight(block.fg[i].wcf,block.fg[i].bsdwcf,nb,lr,bat);
        update_bias(block.fg[i].fb,block.fg[i].bsdfb,lr,bat);

        update_weight(block.wg[i].wiw,block.wg[i].bsdwiw,inn,lr,bat);
        update_weight(block.wg[i].whw,block.wg[i].bsdwhw,nb,lr,bat);
        update_weight(block.wg[i].wcw,block.wg[i].bsdwcw,nb,lr,bat);
        update_bias(block.wg[i].wb,block.wg[i].bsdwb,lr,bat);

        update_weight(block.cl[i].wic,block.cl[i].bsdwic,inn,lr,bat);
        update_weight(block.cl[i].whc,block.cl[i].bsdwhc,nb,lr,bat);
        update_bias(block.cl[i].cb,block.cl[i].bsdcb,lr,bat);
    }
    update_weight(block.weight_co,block.bsdwco,onn,nb,lr,bat);
}

void test(Ilayer &inputl,Olayer &outputl,Block &block){
    int t=inputl.t;
    int onn=outputl.nn;
    int ns=inputl.ns;
    for(int i=0;i<onn;i++){
        outputl.acc[i]=0;
    }
    for(int i=0;i<ns;i++){
        inputl.samth=i;
        outputl.samth=i;
        layer_forward(inputl,outputl,block);
        int ncor=outputl.ncor;
        int lth=outputl.lth;
        outputl.acc[lth]+=(double)ncor/t;

    }
    double accall=0;
    for(int i=0;i<onn;i++){
        outputl.acc[i]=outputl.acc[i]/onn;
        accall+=outputl.acc[i];
    }
    accall/=onn;
    cout<<"acc="<<accall<<endl;
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    int epoch=10000;
    int ns=64;
    double lr=0.0001;
    int batch=16;
    int inum=6;
    int onum=8;
    int nb=45;
    int t=300;
    int test_internal=20;
    Ilayer inputl(ns,t,inum);
    Olayer outputl(ns,t,onum);
    Block block(nb,inum,onum,t);
    vector <int> c(64,0);
    for(int i=0;i<ns;i++){
        c[i]=i;
    }
    vector <int> shun(64,0);
    for(int i=0;i<ns;i++){
        int a=i/8;
        shun[i]=a;
    }
    shuffle(c,shun);
    outputl.zero_count();
    for(int i=0;i<epoch;i++){
        if((i+1)%test_internal==0){
            test(inputl,outputl,block);
        }
        for(int j=0;j<ns;j++){
            int id=c[j];//from 0 to 63
            outputl.samth=id;
            inputl.samth=id;
            layer_forward(inputl,outputl,block);

            layer_backward(inputl,outputl,block);
            outputl.countjia();
            if(outputl.count==batch){
                display(outputl);
                update(inputl,outputl,block,lr,batch);
                outputl.zero_count();
            }
        }
    }
    return a.exec();
}
