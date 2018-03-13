#include <QCoreApplication>
#include<iostream>
#include<stdlib.h>
#include<math.h>
#include<fstream>
#include<time.h>
#include<string.h>
#include<vector>
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
//主类,layer
class layer{
    public:
        int num_node;
        int patum;
        vector <double> weights;
        vector <double> delta;
        vector <double> delta_weight;
        vector <double> delta_weightall;
        int set_numnode(int m){
            num_node=m;
            return 0;
        }
        int get_numnode(){
            return num_node;
        }
        int set_patum(int m){
            patum=m;
            return 0;
        }
        int get_patum(){
            return patum;
        }
        int init_weight(int first,int end,vector <double> &weights);
        int init_delta(int num_node,vector <double> &delta);
        int init_deltaweight(int first,int end,vector <double> &delta_weight);
        int init_deltaweightall(int end,int first,vector <double> &delta_weightall);
        int zero_deltaweight(int end,int first,vector <double> &delta_weight);
        int zero_deltaweightall(int end,int first,vector <double> &delta_weightall);
};
int layer::zero_deltaweight(int end,int first,vector <double> &delta_weight){
    for(int i=0;i<end;i++){
        for(int j=0;j<first;j++){
            delta_weight[i*first+j]=0.0;
        }
    }
    return 0;
}
int layer::zero_deltaweightall(int end,int first,vector <double> &delta_weightall){
    for(int i=0;i<end;i++){
        for(int j=0;j<first;j++){
            delta_weightall[i*first+j]=0.0;
        }
    }
    return 0;
}
int layer::init_delta(int num_node,vector <double> &delta){
        for(int j=0;j<num_node;j++){
            delta.push_back(0.0);
        }
        return 0;
}
int layer::init_deltaweight(int first,    int end,   vector <double> &delta_weight){
    for(int i=0;i<end;i++){
        for(int j=0;j<first;j++){
            delta_weight.push_back(0.0);
        }
    }
    return 0;
}
int layer::init_deltaweightall(int end,int first,vector <double> &delta_weightall){
    for(int i=0;i<end;i++){
        for(int j=0;j<first;j++){
            delta_weightall.push_back(0.0);
        }
    }
    return 0;
}
int layer::init_weight(int first,int end,vector <double> &weights){
    for(int i=0;i<end;i++){
        for(int j=0;j<first;j++){
            double a;
            a=getrand();
            weights.push_back(a);
        }
    }
    return 0;
}
//
//
//这是输入层的类
class Ilayer:public layer{
    public:
        vector <double> input_data;
        int init_inputdata(int num_node,vector <double> &input_data,vector <double> chuan);
        int change_inputdata(int num_node,vector <double> &input_data,vector <double> &chuan);
};
int Ilayer::change_inputdata(int num_node,vector <double> &input_data,vector <double> &chuan){
    for(int i=0;i<num_node;i++){
        input_data[i]=chuan[i];
    }
    return 0;
}
int Ilayer::init_inputdata(int num_node,vector <double> &input_data,vector <double> chuan){
    for(int i=0;i<num_node;i++){
        input_data.push_back(chuan[i]);
    }
    return 0;
}
int change_inputdata(int num_node,vector <double> &input_data,vector <double> chuan){
    for(int i=0;i<num_node;i++){
        input_data[i]=chuan[i];
    }
    return 0;
}
class Hlayer:public layer{
    public:
        vector <double> hidden_val;
        vector <double> sigmoid_val;
        vector <double> bias;
        vector <double> delta_bias;
        vector <double> delta_biasall;
        int init_deltabias(int num_node,vector <double> &delta_bias);
        int init_hiddenval(int num_node,vector <double> &hidden_val);
        int init_sigmoidval(int num_node,vector <double> &sigmod_val);
        int init_bias(int num_node,vector <double> &bias);
        int init_deltabiasall(int num_mode,vector <double> &delta_biasall);
        int zero_deltabias(int num_node,vector <double> &delta_bias);
        int zero_deltabiasall(int num_node,vector <double> &delta_biasall);
};
//初始化biasall
//
int Hlayer::zero_deltabias(int num_node,vector <double> &delta_bias){
    for(int i=0;i<num_node;i++){
        delta_bias[i]=0.0;
    }
    return 0;
}
int Hlayer::zero_deltabiasall(int num_node,vector <double> &delta_biasall){
    for(int i=0;i<num_node;i++){
        delta_biasall[i]=0.0;
    }
    return 0;
}
int Hlayer::init_deltabias(int num_node,vector <double> &delta_bias){
    for(int i=0;i<num_node;i++){
        delta_bias.push_back(0.0);
    }
    return 0;
}
//初始化deltabiasall
//
int Hlayer::init_deltabiasall(int num_node,vector <double> &delta_biasall){
    for(int i=0;i<num_node;i++){
        delta_biasall.push_back(0.0);
    }
    return 0;
}
//初始化bias
//
int Hlayer::init_bias(int num_node,vector <double> &bias){
        for(int j=0;j<num_node;j++){
            double m=getrand();
            bias.push_back(m);
        }
        return 0;
}
//初始化hidden_val
//
int Hlayer::init_hiddenval(int num_node,vector <double> &hidden_val){
        for(int j=0;j<num_node;j++){
            hidden_val.push_back(0.0);
        }
        return 0;
}
//初始化sigmoidval
//
int Hlayer::init_sigmoidval(int num_node,vector <double> &sigmoid_val){
        for(int j=0;j<num_node;j++){
            sigmoid_val.push_back(0.0);
        }
        return 0;
}


//output_layer的类
class Olayer:public layer{
    public:
        vector <double> output;
        vector <double> label;
        int init_output(int num_node,vector <double> &output);
};
int Olayer::init_output(int num_node,vector <double> &output){
        for(int j=0;j<num_node;j++){
            output.push_back(0.0);
        }
        return 0;
}
Olayer init_outputlayer(){
    int onum=2;
    Olayer output_layer;
    output_layer.set_numnode(onum);
    output_layer.init_output(onum,output_layer.output);
    return output_layer;
}
//初始化隐藏层参数
int init_hiddenparam(Hlayer &hidden_layer){
    hidden_layer.init_hiddenval(hidden_layer.num_node,hidden_layer.hidden_val);
    hidden_layer.init_sigmoidval(hidden_layer.num_node,hidden_layer.sigmoid_val);
    hidden_layer.init_bias(hidden_layer.num_node,hidden_layer.bias);
    hidden_layer.init_deltabias(hidden_layer.num_node,hidden_layer.delta_bias);
    hidden_layer.init_deltabiasall(hidden_layer.num_node,hidden_layer.delta_biasall);
    return 0;
}
int zero_hiddenparam(Hlayer &hidden_layer){
    for(int i=0;i<hidden_layer.num_node;i++){
        hidden_layer.delta_bias[i]=0.0;
    }
    for(int i=0;i<hidden_layer.num_node;i++){
        hidden_layer.delta_biasall[i]=0.0;
    }
    for(int i=0;i<hidden_layer.num_node;i++){
        hidden_layer.delta[i]=0.0;
    }
    return 0;
}
int zero_outputdelta(Olayer &output_layer){
    for(int i=0;i<output_layer.num_node;i++){
        output_layer.delta[i]=0.0;
    }
    return 0;
}
//input_layer和hidden_layer之间的连接
int data_trainIH(int first,
        int end,
        vector <double> &data,
        vector <double> &weights,
        vector <double> &hidden_val,
        vector <double> &sigmoid_val,
        vector <double> &bias)
{
        for(int j=0;j<end;j++){//相当于后一个节点的个数,weight节点
            hidden_val[j]=0.0;
            for(int k=0;k<first;k++){//相当于前一个节点的个数,数据节点点
                hidden_val[j]=hidden_val[j]+data[k]*weights[j*first+k];
            }
        hidden_val[j]=hidden_val[j]+bias[j];
        sigmoid_val[j]=sigmoid(hidden_val[j]);
        }
    return 0;
}
//隐藏层和输出层之间的连接
int data_trainHO(int first,
        int end,
        vector <double> &sig1,
        vector <double> &weights,
        vector <double> &output)
{
        for(int j=0;j<end;j++){//相当于后一个节点的个数,weight节点
        output[j]=0.0;
            for(int k=0;k<first;k++){//相当于前一个节点的个数,数据节点点

                output[j]=output[j]+sig1[k]*weights[j*first+k];
            }
        output[j]=sigmoid(output[j]);//就是yn
    }

    return 0;
}
int cal_delta(int num_node,
        vector <double> &delta,
        vector <double> &output,
        vector <double> &label)
{
        for(int j=0;j<num_node;j++){
            delta[j]=output[j]-label[j];
        }
        return 0;
}
int delta_bias_weight(int first,
        int end,
        vector <double> &deltafirst,
        vector <double> &sigmoid_val,
        vector <double> &weights,
        vector <double> &deltalast,
        vector <double> &delta_bias,
        vector <double> &delta_weight,
        vector <double> &delta_weightall,
        vector <double> &delta_biasall,
        vector <double> &output)
{
        for(int j=0;j<end;j++){
            for(int k=0;k<first;k++){
                deltalast[j]=weights[k*end+j]*deltafirst[k];//在这里deltalast又给该层的delta赋值.
                delta_weight[k*end+j]=sigmoid_val[j]*deltafirst[k]*fsigmoid(sigmoid_val[j]);
                delta_weightall[k*end+j]+=delta_weight[k*end+j];
            }
            delta_bias[j]=fsigmoid(sigmoid_val[j])*deltalast[j];//在这里
            delta_biasall[j]+=delta_bias[j];
    }
    return 0;

}
//计算delta_weight
//
int delta_weight(//第几个patum
        int first,
        int end,
        vector <double> &delta,
        vector <double> &data,
        vector <double> &delta_weight,
        vector <double> &delta_weightall,
        vector <double> &sigmoid_val)
{
    for(int j=0;j<end;j++){
        for(int k=0;k<first;k++){
            delta_weight[k*end+j]=data[j]*delta[k]*sigmoid_val[k];
            delta_weightall[k*end+j]+=delta_weight[k*end+j];
        }
    }
    return 0;
}

int weight_changeIH(double lr,
        int patum,
        Ilayer &input_layer)
{
    int m=input_layer.weights.size();
    for(int i=0;i<m;i++){
        input_layer.weights[i]=input_layer.weights[i]-lr*(input_layer.delta_weightall[i])/patum;
    }
    return 0;
}
int weight_changeHO(double lr,
        int patum,
        Hlayer &hidden_layer){
    int m=hidden_layer.weights.size();
    for(int i=0;i<m;i++){
        hidden_layer.weights[i]=hidden_layer.weights[i]-lr*hidden_layer.delta_weightall[i]/patum;
    }
    cout<<endl;
    return 0;
}
//计算bias_change
//
int bias_change(double lr,
        int patum,
        Hlayer &hidden_layer)
{
    int m=hidden_layer.num_node;
    for(int i=0;i<m;i++){
        hidden_layer.bias[i]=hidden_layer.bias[i]-lr*(hidden_layer.delta_biasall[i]/patum);
    }
    return 0;
}
//写入权值到文件中去

void writeweight(layer layerparam){
    ofstream ofile;
    ofile.open("/home/zzq/mlpdebug/param.txt",ios::app);
    int m=layerparam.weights.size();
    for(int i=0;i<m;i++){
        ofile<<layerparam.weights[i]<<endl;
    }
    ofile.close();
    cout<<endl;
}
//写入bias到文件中去
//*

template <typename Dtype>
int writebias(Dtype layerparam){
    ofstream ofile;
    ofile.open("/home/zzq/code/mlpdebug/param.txt",ios::app);
    int m=layerparam.bias.size();
    for(int i=0;i<m;i++){
        ofile<<layerparam.bias[i]<<endl;
    }
    ofile.close();
    return 0;
}

double display(Olayer &output_layer){
    double error=0.0;
    for(int i=0;i<output_layer.num_node;i++){
        error+=(output_layer.output[i]-output_layer.label[i])*(output_layer.output[i]-output_layer.label[i]);
    }
    return error;
}

int test(Ilayer input_layer,Hlayer hidden_layer,Olayer output_layer){
    ifstream ifile;
    ifile.open("/home/zzq/code/mlpdebug/val1");
    int val_patum=69;
    int inum=input_layer.num_node;
    double data[69][30];
    for(int i=0;i<val_patum;i++){
        for(int j=0;j<inum;j++){
            ifile>>data[i][j];
        }
    }
    ifile.close();
    ifstream lfile;
    lfile.open("/home/zzq/code/mlpdebug/bestval");
    int label[69];
    for(int i=0;i<69;i++){
        lfile>>label[i];
    }
    lfile.close();
    vector <double> chuan;
    for(int i=0;i<inum;i++){
        chuan.push_back(data[0][i]);
    }
    int correct=0;
    for(int m=0;m<69;m++){
        for(int po=0;po<inum;po++){
            chuan[po]=data[m][po];
        }
        input_layer.change_inputdata(inum,input_layer.input_data,chuan);
        data_trainIH(input_layer.num_node,
                hidden_layer.num_node,
                input_layer.input_data,
                input_layer.weights,
                hidden_layer.hidden_val,
                hidden_layer.sigmoid_val,
                hidden_layer.bias);
        data_trainHO(hidden_layer.num_node,
                output_layer.num_node,
                hidden_layer.sigmoid_val,
                hidden_layer.weights,
                output_layer.output);
        int max=-10000;
        int flag;
        for(int k=0;k<output_layer.num_node;k++){
            if(max<output_layer.output[k]){
                max=output_layer.output[k];
                flag=k;
            }
        }
        if(flag==label[m]){
            correct+=1;
        }
    }
    double acc;
    double dc;
    dc=(double)correct;
    acc=dc/val_patum;
    return 0;

}//主函数
int traintest(Ilayer &input_layer,Hlayer &hidden_layer,Olayer &output_layer){
    ifstream ifile;
    double error=0.0;
    int train_patum=500;
    ifile.open("/home/zzq/code/mlpdebug/data1");
    int inum=input_layer.num_node;
    double data[500][30];
    for(int i=0;i<train_patum;i++){
        for(int j=0;j<inum;j++){
            ifile>>data[i][j];
        }
    }
    ifile.close();
    ifstream lfile;
    lfile.open("/home/zzq/code/mlpdebug/bestdata");
    int trainlabel[500];
    for(int i=0;i<500;i++){
        lfile>>trainlabel[i];
    }
    lfile.close();
    vector <double> chuan;
    for(int i=0;i<inum;i++){
        chuan.push_back(data[0][i]);
    }
    int correct=0;
    for(int m=0;m<500;m++){
        for(int po=0;po<inum;po++){
            chuan[po]=data[m][po];
        }
        input_layer.change_inputdata(inum,input_layer.input_data,chuan);
        data_trainIH(input_layer.num_node,
                hidden_layer.num_node,
                input_layer.input_data,
                input_layer.weights,
                hidden_layer.hidden_val,
                hidden_layer.sigmoid_val,
                hidden_layer.bias);
        data_trainHO(hidden_layer.num_node,
                output_layer.num_node,
                hidden_layer.sigmoid_val,
                hidden_layer.weights,
                output_layer.output);
        int max=-10000;
        int flag;
        for(int k=0;k<output_layer.num_node;k++){
            if(max<output_layer.output[k]){
                max=output_layer.output[k];
                flag=k;
            }
        }
        if(flag!=trainlabel[m]){
            correct+=1;
        }
    double dou=display(output_layer);
    error+=dou;
    }
    error=error/train_patum;
    double acc;
    double dc;
    dc=(double)correct;
    acc=dc/train_patum;
    cout<<"\t"<<"acc="<<acc<<endl; // 训练的正确率
    return 0;
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    int epoch=5;
    int patum=500;
    double lr=0.01;
    Ilayer input_layer;
    int inum=30;
    input_layer.set_numnode(inum);
    ifstream ifile;
    ifile.open("/home/zzq/code/mlpdebug/data1");
    double fileparam[500][30];
    for(int i=0;i<patum;i++){
        for(int j=0;j<inum;j++){
            ifile>>fileparam[i][j];
        }
    }
    ifile.close();
    vector <double> chuan;
        for(int j=0;j<inum;j++){
            chuan.push_back(fileparam[0][j]);
        }
    input_layer.init_inputdata(inum,input_layer.input_data,chuan);
    Hlayer hidden_layer;
        int hnum=300;
        int num_hiddenlayer=1;
        hidden_layer.set_numnode(hnum);
    int onum=2;
    Olayer output_layer;
    output_layer.set_numnode(onum);
    output_layer.init_output(onum,output_layer.output);//初始化output,全设为0
    output_layer.init_delta(onum,output_layer.delta);//初始化delta
    hidden_layer.init_delta(hnum,hidden_layer.delta);
    ifstream lfile;
    lfile.open("/home/zzq/code/mlpdebug/bestdata");
    int flag[500];
    for(int i=0;i<patum;i++){
        lfile>>flag[i];
    }
    lfile.close();
    output_layer.label.push_back(1.0);//正确
    output_layer.label.push_back(0.0);
//初始化权值
    input_layer.init_weight(input_layer.num_node,
            hidden_layer.num_node,
            input_layer.weights);
    hidden_layer.init_weight(hidden_layer.num_node,
            output_layer.num_node,
            hidden_layer.weights);
    init_hiddenparam(hidden_layer);
    hidden_layer.init_deltaweight(hidden_layer.num_node,
            output_layer.num_node,
            hidden_layer.delta_weight);
    hidden_layer.init_deltaweightall(output_layer.num_node,
            hidden_layer.num_node,
            hidden_layer.delta_weightall);
    input_layer.init_deltaweight(input_layer.num_node,
            hidden_layer.num_node,
            input_layer.delta_weight);
    input_layer.init_deltaweightall(hidden_layer.num_node,
            input_layer.num_node,
            input_layer.delta_weightall);

//前向计算
    for(int i=0;i<epoch;i++){
        for(int j=0;j<patum;j++){
        for(int l=0;l<inum;l++){
            chuan[l]=fileparam[j][l];
        }
        input_layer.change_inputdata(inum,input_layer.input_data,chuan);//改变数据的
        if(flag[j]==0){
            output_layer.label[0]=1.0;
            output_layer.label[1]=0.0;
        }
        else{
            output_layer.label[0]=0.0;
            output_layer.label[1]=1.0;

        }
        //数据读入没有问题,输入层数据没有问题
        data_trainIH(
                input_layer.num_node,
                hidden_layer.num_node,
                input_layer.input_data,
                input_layer.weights,
                hidden_layer.hidden_val,
                hidden_layer.sigmoid_val,
                hidden_layer.bias);
        //计算隐藏层与输出层之间的数据传输
        data_trainHO(hidden_layer.num_node,
                output_layer.num_node,
                hidden_layer.sigmoid_val,
                hidden_layer.weights,
                output_layer.output);
//后向计算
        //计算残差
        cal_delta(output_layer.num_node,
                output_layer.delta,
                output_layer.output,
                output_layer.label);
        delta_bias_weight(output_layer.num_node,
                hidden_layer.num_node,
                output_layer.delta,
                hidden_layer.sigmoid_val,
                hidden_layer.weights,
                hidden_layer.delta,
                hidden_layer.delta_bias,
                hidden_layer.delta_weight,
                hidden_layer.delta_weightall,
                hidden_layer.delta_biasall,
                output_layer.output);//计算输出层与隐藏层之间的delta_weight,delta__bias

        delta_weight(
                hidden_layer.num_node,
                input_layer.num_node,
                hidden_layer.delta,
                input_layer.input_data,
                input_layer.delta_weight,
                input_layer.delta_weightall,
                hidden_layer.sigmoid_val);//改变隐藏层和输入层之间的连接
        }
        weight_changeIH(lr,patum,input_layer);
        weight_changeHO(lr,patum,hidden_layer);
        bias_change(lr,patum,hidden_layer);

    for(int p=0;p<num_hiddenlayer;p++){
        zero_hiddenparam(hidden_layer);
    }
    zero_outputdelta(output_layer);
    input_layer.zero_deltaweight(input_layer.num_node,
            hidden_layer.num_node,
            input_layer.delta_weight);
    input_layer.zero_deltaweightall(input_layer.num_node,
            hidden_layer.num_node,
            input_layer.delta_weightall);//input层

    hidden_layer.zero_deltabias(hidden_layer.num_node,
            hidden_layer.delta_bias);
    hidden_layer.zero_deltabiasall(hidden_layer.num_node,
            hidden_layer.delta_biasall);
      hidden_layer.zero_deltaweight(hidden_layer.num_node,
            output_layer.num_node,
            hidden_layer.delta_weight);
    hidden_layer.zero_deltaweightall(hidden_layer.num_node,
            output_layer.num_node,
            hidden_layer.delta_weightall);
    traintest(input_layer,hidden_layer,output_layer);
    }

     test(input_layer,hidden_layer,output_layer);
    ofstream ofile;
    ofile.open("/home/zzq/code/mlpdebug/param.txt");
    ofile<<epoch<<endl;
    ofile<<patum<<endl;
    ofile<<lr<<endl;
    ofile<<num_hiddenlayer<<endl;
    ofile<<input_layer.num_node<<endl;
    ofile.close();
    writeweight(input_layer);
    writeweight(hidden_layer);
    writebias(hidden_layer);

    return a.exec();
}
