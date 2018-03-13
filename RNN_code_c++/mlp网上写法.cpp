#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_NO_OF_LAYERS 3
#define MAX_NO_OF_INPUTS 2
#define MAX_NO_OF_NEURONS 10
#define MAX_NO_OF_WEIGHTS 31
#define MAX_NO_OF_OUTPUTS 2
//函数声明
void createNet( int, int *, int *, char *, double *, int );
void feedNetInputs(double *);
void updateNetOutput(void);
double *getOutputs();
void trainNet ( double, double, int, double * );
void applyBatchCumulations( double, double );
int loadNet(char *);
int saveNet(char *);
double getRand();

struct neuron{
    double *output;
    double threshold;
    double oldThreshold;
    double batchCumulThresholdChange;
    char axonFamily;
    double *weights;
    double *oldWeights;
    double *netBatchCumulWeightChanges;
    int noOfInputs;
    double *inputs;
    double actFuncFlatness;
    double *error;
};

struct layer {
    int noOfNeurons;
    struct neuron * neurons;
};
//两个嵌套的结构体
static struct neuralNet {
    int noOfInputs;
    double *inputs;
    double *outputs;
    int noOfLayers;
    struct layer *layers;
    int noOfBatchChanges;
} theNet;//初始化一个神经网络

double getRand() {

    return (( (double)rand() * 2 ) / ( (double)RAND_MAX + 1 ) ) - 1;

}

static struct neuron netNeurons[MAX_NO_OF_NEURONS];//neuron是个结构体
static double netInputs[MAX_NO_OF_INPUTS];
static double netNeuronOutputs[MAX_NO_OF_NEURONS];
static double netErrors[MAX_NO_OF_NEURONS];
static struct layer netLayers[MAX_NO_OF_LAYERS];//layer也是个结构体
static double netWeights[MAX_NO_OF_WEIGHTS];
static double netOldWeights[MAX_NO_OF_WEIGHTS];
static double netBatchCumulWeightChanges[MAX_NO_OF_WEIGHTS];

void createNet( int noOfLayers, int *noOfNeurons, int *noOfInputs, char *axonFamilies, double *actFuncFlatnesses, int initWeights ) {

    int i, j, counter, counter2, counter3, counter4;
    int totalNoOfNeurons, totalNoOfWeights;

    theNet.layers = netLayers;//使用静态的已经生成的数据去给thenet中元素赋值
    theNet.noOfLayers = noOfLayers;
    theNet.noOfInputs = noOfInputs[0];//指针中的第一个数赋值
    theNet.inputs = netInputs;//用数组给指针赋值,用数组给指针赋值,等于其第一个元素

    totalNoOfNeurons = 0;
    for(i = 0; i < theNet.noOfLayers; i++) {
        totalNoOfNeurons += noOfNeurons[i];//总的神经元个数
    }
    for(i = 0; i < totalNoOfNeurons; i++) { netNeuronOutputs[i] = 0; }

    totalNoOfWeights = 0;
    for(i = 0; i < theNet.noOfLayers; i++) {//层数
        totalNoOfWeights += noOfInputs[i] * noOfNeurons[i];
    }

    counter = counter2 = counter3 = counter4 = 0;
    for(i = 0; i < theNet.noOfLayers; i++) {
        for(j = 0; j < noOfNeurons[i]; j++) {
            if(i == theNet.noOfLayers-1 && j == 0) { // beginning of the output layer
                theNet.outputs = &netNeuronOutputs[counter];//最后一层的各个神经元
            }
            netNeurons[counter].output = &netNeuronOutputs[counter];
            netNeurons[counter].noOfInputs = noOfInputs[i];//这是结构体?还是指针
            netNeurons[counter].weights = &netWeights[counter2];//这是因为netweight是一个数组
            netNeurons[counter].netBatchCumulWeightChanges = &netBatchCumulWeightChanges[counter2];
            netNeurons[counter].oldWeights = &netOldWeights[counter2];
            netNeurons[counter].axonFamily = axonFamilies[i];
            netNeurons[counter].actFuncFlatness = actFuncFlatnesses[i];
            if ( i == 0) {
                netNeurons[counter].inputs = netInputs;
            }
            else {
                netNeurons[counter].inputs = &netNeuronOutputs[counter3];
            }
            netNeurons[counter].error = &netErrors[counter];
            counter2 += noOfInputs[i];
            counter++;
        }
        netLayers[i].noOfNeurons = noOfNeurons[i];
        netLayers[i].neurons = &netNeurons[counter4];
        if(i > 0) {
            counter3 += noOfNeurons[i-1];
        }
        counter4 += noOfNeurons[i];
    }

    // initialize weights and thresholds
     if ( initWeights == 1 ) {
        for( i = 0; i < totalNoOfNeurons; i++) { netNeurons[i].threshold = getRand(); }
        for( i = 0; i < totalNoOfWeights; i++) { netWeights[i] = getRand(); }
        for( i = 0; i < totalNoOfWeights; i++) { netOldWeights[i] = netWeights[i]; }
        for( i = 0; i < totalNoOfNeurons; i++) { netNeurons[i].oldThreshold = netNeurons[i].threshold; }
    }

    // initialize batch values
    for( i = 0; i < totalNoOfNeurons; i++) { netNeurons[i].batchCumulThresholdChange = 0; }
    for( i = 0; i < totalNoOfWeights; i++) { netBatchCumulWeightChanges[i] = 0; }
    theNet.noOfBatchChanges = 0;

}

void feedNetInputs(double *inputs) {
     int i;
     for ( i = 0; i < theNet.noOfInputs; i++ ) {
        netInputs[i] = inputs[i];
     }
}

static void updateNeuronOutput(struct neuron * myNeuron) {

    double activation = 0;
    int i;

    for ( i = 0; i < myNeuron->noOfInputs; i++) {
        activation += myNeuron->inputs[i] * myNeuron->weights[i];
    }
    activation += -1 * myNeuron->threshold;
    double temp;
    switch (myNeuron->axonFamily) {
        case 'g': // logistic
            temp = -activation / myNeuron->actFuncFlatness;
            /* avoid overflow */
            if ( temp > 45 ) {
                *(myNeuron->output) = 0;
            }
            else if ( temp < -45 ) {
                *(myNeuron->output) = 1;
            }
            else {
                *(myNeuron->output) = 1.0 / ( 1 + exp( temp ));
            }
            break;
        case 't': // tanh
            temp = -activation / myNeuron->actFuncFlatness;
            /* avoid overflow */
            if ( temp > 45 ) {
                *(myNeuron->output) = -1;
            }
            else if ( temp < -45 ) {
                *(myNeuron->output) = 1;
            }
            else {
                *(myNeuron->output) = ( 2.0 / ( 1 + exp( temp ) ) ) - 1;
            }
            break;
        case 'l': // linear
            *(myNeuron->output) = activation;
            break;
        default:
            break;
    }

}

void updateNetOutput( ) {

    int i, j;

    for(i = 0; i < theNet.noOfLayers; i++) {
        for( j = 0; j < theNet.layers[i].noOfNeurons; j++) {
            updateNeuronOutput(&(theNet.layers[i].neurons[j]));//指针加取址符
        }
    }

}

static double derivative (struct neuron * myNeuron) {//导函数,激活函数的导函数，针对不同激活函数的导函数

    double temp;
    switch (myNeuron->axonFamily) {//激活函数的不同类型
        case 'g': // logistic
            temp = ( *(myNeuron->output) * ( 1.0 - *(myNeuron->output) ) ) / myNeuron->actFuncFlatness; break;
        case 't': // tanh
            temp = ( 1 - pow( *(myNeuron->output) , 2 ) ) / ( 2.0 * myNeuron->actFuncFlatness ); break;
        case 'l': // linear
            temp = 1; break;
        default:
            temp = 0; break;
    }
    return temp;

}

// learningRate and momentumRate will have no effect if batch mode is 'on'
void trainNet ( double learningRate, double momentumRate, int batch, double *outputTargets ) {

    int i,j,k;
    double temp;
    struct layer *currLayer, *nextLayer;

     // calculate errors
    for(i = theNet.noOfLayers - 1; i >= 0; i--) {
        currLayer = &theNet.layers[i];
        if ( i == theNet.noOfLayers - 1 ) { // output layer
            for ( j = 0; j < currLayer->noOfNeurons; j++ ) {
                *(currLayer->neurons[j].error) = derivative(&currLayer->neurons[j]) * ( outputTargets[j] - *(currLayer->neurons[j].output));//最后一层的误差等于导函数乘以output-label
            }
        }
        else { // other layers
            nextLayer = &theNet.layers[i+1];//用一个指针来改变指向，指针和取址符的联合使用
            for ( j = 0; j < currLayer->noOfNeurons; j++ ) {//一层之间的网络节点
                temp = 0;
                for ( k = 0; k < nextLayer->noOfNeurons; k++ ) {
                    temp += *(nextLayer->neurons[k].error) * nextLayer->neurons[k].weights[j];
                }
                *(currLayer->neurons[j].error) = derivative(&currLayer->neurons[j]) * temp;//每一层的error都事先计算好，都储存在neuron中
            }
        }
    }

    // update weights n thresholds
    double tempWeight;
    for(i = theNet.noOfLayers - 1; i >= 0; i--) {
        currLayer = &theNet.layers[i];
        for ( j = 0; j < currLayer->noOfNeurons; j++ ) {

            // thresholds
            if ( batch == 1 ) {
                    currLayer->neurons[j].batchCumulThresholdChange += *(currLayer->neurons[j].error) * -1;//batch为1的情况
            }
            else {
                tempWeight = currLayer->neurons[j].threshold;
                currLayer->neurons[j].threshold += ( learningRate  *     *(currLayer->neurons[j].error) * -1 ) + ( momentumRate * ( currLayer->neurons[j].threshold - currLayer->neurons[j].oldThreshold ) );//中间两个*的地方第二个*是指针
                currLayer->neurons[j].oldThreshold = tempWeight;//更新阈值
            }
//阈值的更新方式是学习率乘以error乘以-1再加上momentum乘以两次阈值的差
            // weights
            if ( batch == 1 ) {
                for( k = 0; k < currLayer->neurons[j].noOfInputs; k++ ) {
                    currLayer->neurons[j].netBatchCumulWeightChanges[k] +=  *(currLayer->neurons[j].error) * currLayer->neurons[j].inputs[k];//error乘以input就是delta_weight,这里没有乘上学习率
                }
            }
            else {
                for( k = 0; k < currLayer->neurons[j].noOfInputs; k++ ) {
                    tempWeight = currLayer->neurons[j].weights[k];
                    currLayer->neurons[j].weights[k] += ( learningRate * *(currLayer->neurons[j].error) * currLayer->neurons[j].inputs[k] ) + ( momentumRate * ( currLayer->neurons[j].weights[k] - currLayer->neurons[j].oldWeights[k] ) );//
                    currLayer->neurons[j].oldWeights[k] = tempWeight;//更新权重
                }
            }
//权重的更新方式是学习率乘以error乘以input再加上动量乘以两次权重的差
        }
    }

    if(batch == 1) {
        theNet.noOfBatchChanges++;//更新次数
    }

}

void applyBatchCumulations( double learningRate, double momentumRate ) {

    int i,j,k;
    struct layer *currLayer;
    double tempWeight;

    for(i = theNet.noOfLayers - 1; i >= 0; i--) {
        currLayer = &theNet.layers[i];
        for ( j = 0; j < currLayer->noOfNeurons; j++ ) {
            // thresholds
            tempWeight = currLayer->neurons[j].threshold;
            currLayer->neurons[j].threshold += ( learningRate * ( currLayer->neurons[j].batchCumulThresholdChange / theNet.noOfBatchChanges ) ) + ( momentumRate * ( currLayer->neurons[j].threshold - currLayer->neurons[j].oldThreshold ) );
            currLayer->neurons[j].oldThreshold = tempWeight;
            currLayer->neurons[j].batchCumulThresholdChange = 0;
            // weights
            for( k = 0; k < currLayer->neurons[j].noOfInputs; k++ ) {
                tempWeight = currLayer->neurons[j].weights[k];
                currLayer->neurons[j].weights[k] += ( learningRate * ( currLayer->neurons[j].netBatchCumulWeightChanges[k] / theNet.noOfBatchChanges ) ) + ( momentumRate * ( currLayer->neurons[j].weights[k] - currLayer->neurons[j].oldWeights[k] ) );//都加上了一个动量乘以权重的变化
                currLayer->neurons[j].oldWeights[k] = tempWeight;//以前的权重就等于现在的
                currLayer->neurons[j].netBatchCumulWeightChanges[k] = 0;//清零处理，每一个节点的都清零
            }
        }//这是每一层
    }

    theNet.noOfBatchChanges = 0;

}

double *getOutputs() {

    return theNet.outputs;//返回的是指针

}

int loadNet(char *path) {

    int tempInt; double tempDouble; char tempChar;
    int i, j, k;

    int noOfLayers;
    int noOfNeurons[MAX_NO_OF_LAYERS];
    int noOfInputs[MAX_NO_OF_LAYERS];
    char axonFamilies[MAX_NO_OF_LAYERS];
    double actFuncFlatnesses[MAX_NO_OF_LAYERS];

    FILE *inFile;

    if(!(inFile = fopen(path, "rb")))//加载文件的路径
    return 1;

    fread(&tempInt,sizeof(int),1,inFile);//读取文件数据的一个操作
    noOfLayers = tempInt;//层的个数

    for(i = 0; i < noOfLayers; i++) {

        fread(&tempInt,sizeof(int),1,inFile);
        noOfNeurons[i] = tempInt;//加载数据

        fread(&tempInt,sizeof(int),1,inFile);
        noOfInputs[i] = tempInt;

        fread(&tempChar,sizeof(char),1,inFile);
        axonFamilies[i] = tempChar;//激活函数的类型

        fread(&tempDouble,sizeof(double),1,inFile);//从文件中读取数据
        actFuncFlatnesses[i] = tempDouble;//一开始就初始化，从文件中初始化

    }

    createNet(noOfLayers, noOfNeurons, noOfInputs, axonFamilies, actFuncFlatnesses, 0);//构建网络，初始化网络

    // now the weights
    for(i = 0; i < noOfLayers; i++) {
        for (j = 0; j < noOfNeurons[i]; j++) {
            fread(&tempDouble,sizeof(double),1,inFile);
            theNet.layers[i].neurons[j].threshold = tempDouble;//从文件中读取每个节点的阈值
            for ( k = 0; k < noOfInputs[i]; k++ ) {
                fread(&tempDouble,sizeof(double),1,inFile);
                theNet.layers[i].neurons[j].weights[k] = tempDouble;//读取输入值
            }
        }
    }

    fclose(inFile);

    return 0;

}

int main() {

    double inputs[MAX_NO_OF_INPUTS];//inputs[2]
    double outputTargets[MAX_NO_OF_OUTPUTS];//个数为2

    /* determine layer paramaters */
    int noOfLayers = 3; // input layer excluded
    int noOfNeurons[] = {5,3,2};
    int noOfInputs[] = {2,5,3};
    char axonFamilies[] = {'g','g','t'};
    double actFuncFlatnesses[] = {1,1,1};

    createNet(noOfLayers, noOfNeurons, noOfInputs, axonFamilies, actFuncFlatnesses, 1);

    /* train it using batch method */
    int i;
    double tempTotal;
    int counter = 0;
    for(i = 0; i < 100000; i++) {//训练阶段
        inputs[0] = getRand();//怎么每一次都得到一个随机值
        inputs[1] = getRand();
        tempTotal = inputs[0] + inputs[1];
        feedNetInputs(inputs);
        updateNetOutput();
        outputTargets[0] = (double)sin(tempTotal);
        outputTargets[1] = (double)cos(tempTotal);
        /* train using batch training ( don't update weights, just cumulate them ) */
        trainNet(0, 0, 1, outputTargets);
        counter++;
        /* apply batch changes after 1000 loops use .8 learning rate and .8 momentum */
        if(counter == 100) { applyBatchCumulations(.8,.8); counter = 0;}//batch为100，如果counter达到了100就更新一次
    }

    /* test it */
    double *outputs;
    printf("Sin Target \t Output \t Cos Target \t Output\n");
    printf("---------- \t -------- \t ---------- \t --------\n");
    for(i = 0; i < 50; i++) {
        inputs[0] = getRand();
        inputs[1] = getRand();
        tempTotal = inputs[0] + inputs[1];
        feedNetInputs(inputs);
        updateNetOutput();
        outputs = getOutputs();
        printf( "%f \t %f \t %f \t %f \n", sin(tempTotal), outputs[0], cos(tempTotal), outputs[1]);//测试阶段
    }
    getch();
    return 0;

}
