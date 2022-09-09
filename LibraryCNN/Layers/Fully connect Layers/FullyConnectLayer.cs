using LibraryCNN.Other;
using System;
using System.Linq;

namespace LibraryCNN
{
    [Serializable]
    public class FullyConnectLayer : AbLayer
    {
        protected int input = 5;
        protected TypeActivation typeBackA = TypeActivation.Sigmoid;
        protected internal Tensor Neurons { get; set; }//Значения нейронов слоя
        public virtual int Input //Кол-во нейронов
        { 
            get { return input; } 
            set { if (value > 0) { input = value; OnParamsChanged(new EventChangedParams()); } }
        }
        protected int Next //Кол-во нейронов след.слоя
        {
            get 
            {
                int index = Network.layersFully.Layer.IndexOf(this);
                return index < Network.layersFully.Layer.Count - 1 ? Network.layersFully.Layer[index + 1].Input : 0;
            }
        }
        protected TypeActivation TypeForwardA //Тип активации слоя
        {
            get
            {
                int index = Network.layersFully.Layer.IndexOf(this);
                return index < Network.layersFully.Layer.Count - 1 ? Network.layersFully.Layer[index + 1].TypeBackA : TypeActivation.Input;
            }
        }
        protected internal double Loss { get; protected set; }//Функция потерь
        public virtual TypeActivation TypeBackA //Тип производной для обратного прохода
        { 
            get { return typeBackA; } 
            set { typeBackA = value; OnParamsChanged(new EventChangedParams()); }
        }
        public FullyConnectLayer() { ParamsChanged += FullyConnectLayer_ParamsChanged; }
        protected void FullyConnectLayer_ParamsChanged(object sender, EventChangedParams e)
        {
            Network.InitializationFully();
        }
        protected Tensor Corrections { get; set; }//Корректировки весов
        protected Tensor Bias { get; set; }//Массив смещений
        protected Matrix Weights { get; set; }//веса
        public void DropOut(double p)
        {
            for(int n = 0; n < Output.FullSize; n++)
            {
                Output[n, 0, 0, p] = true;
            }
        }
        public void Unlocked(double p)
        {
            for (int n = 0; n < Output.FullSize; n++)
            {
                Output[n, 0, 0, p] = false;
            }
        }
        //Прямой проход(Forward)
        public override Tensor Forward(in Tensor input)
        {
            Neurons = input;
            for (int w1 = 0; w1 < Weights.m; w1++)
            {
                double Sum = 0;
                for (int w2 = 0; w2 < Weights.n; w2++)
                {
                    Sum += Neurons[w2, 0, 0] * Weights[w2, w1];
                }
                Output[w1, 0, 0] = FuncActivations.Activation(TypeForwardA, Sum, Neurons, Output, w1) * (1 / (1 - Network.Probability));
            }
            return Output;
        }
        //Обратный проход(BackPropagation)
        public override void BackPropagation(in Tensor delta, in int Right, in double E, in double A)//int Right, ForwardLayer PreDelta, double E, double A)
        {
            double dout;
            Tensor DeltaNext = new Tensor(Neurons.SizeZ, Neurons.SizeX, Neurons.SizeY);
            for (int i = 0; i < Weights.m; i++)
            {
                for (int j = 0; j < Weights.n; j++)
                {
                    dout = FuncActivations.Deactivation(TypeBackA, Neurons, Neurons[j, 0, 0]);
                    DeltaNext[j, 0, 0] += dout * (Weights[j, i] * delta[i, 0, 0]);//Градиент для следующего слоя 
                    double GRADw = Neurons[j, 0, 0] * delta[i, 0, 0];//Градиент данного слоя
                    Corrections[j, 0, 0] = E * GRADw + A * Corrections[j, 0, 0]; //Посчет обновления весов по градиенту и коф.обучения
                    Weights[j, i] += Corrections[j, 0, 0];//Обновление весов
                }
            }
            DeltaList = DeltaNext;
        }
        public override void ReSizer(int inputNew)
        {
            Weights = new Matrix(inputNew, Weights.m);//Кол-во весов
            Neurons = new Tensor(inputNew, 1, 1);//Кол-во нейронов на этом слое
            Corrections = new Tensor(inputNew, 1, 1);//Кол-во корректировок
            RandomParams();//Заполнение весов
        }
        //Другое
        public virtual void Initialization()
        {
            //Neurons = new Tensor(Input, 1, 1);
            Corrections = new Tensor(Input, 1, 1);//Кол-во корректировок
            Weights = new Matrix(Input, Next);//Кол-во весов
            Output = new Tensor(Next, 1, 1);//Кол-во выходных значений
            RandomParams();
        }
        protected override void RandomParams()//Генерация рандомных весов
        {
            for (int i = 0; i < Weights.n; i++)
            {
                for (int j = 0; j < Weights.m; j++)
                {
                    Weights[i, j] = Converter.RandomValues();
                }
            }
        }
    }
}
