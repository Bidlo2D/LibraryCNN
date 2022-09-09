using LibraryCNN.Other;
using System;
namespace LibraryCNN
{
    [Serializable]
    public class FullyConnectBias : FullyConnectLayer
    {
        public override Tensor Forward(in Tensor input)
        {
            Neurons = input;
            //Output = new Tensor(Weights.m, 1, 1);
            for (int w1 = 0; w1 < Weights.m; w1++)
            {
                double Sum = Bias[w1, 0, 0];// + смещение
                for (int w2 = 0; w2 < Weights.n; w2++)
                {
                    Sum += Neurons[w2, 0, 0] * Weights[w2, w1];
                }
                Output[w1, 0, 0] = FuncActivations.Activation(TypeForwardA, Sum, Neurons, Output, w1) * (1 / (1 - Network.Probability));
            }
            return Output;
        }
        public override void BackPropagation(in Tensor delta, in int Right, in double E, in double A)
        {
            double dout = 0.0;
            Tensor DeltaNext = new Tensor(Neurons.SizeZ, Neurons.SizeX, Neurons.SizeY);
            for (int i = 0; i < Weights.m; i++)
            {
                double df = 0;
                for (int j = 0; j < Weights.n; j++)
                {
                    dout = FuncActivations.Deactivation(TypeBackA, Neurons, Neurons[j, 0, 0]);//производная функции активации
                    df += dout * delta[i, 0, 0];//градиент для смещений
                    DeltaNext[j, 0, 0] += dout * (Weights[j, i] * delta[i, 0, 0]);//Градиент для следующего слоя 
                    double GRADw = Neurons[j, 0, 0] * delta[i, 0, 0];//Градиент данного слоя
                    Corrections[j, 0, 0] = E * GRADw + A * Corrections[j, 0, 0]; //Посчет обновления весов по градиенту и коф.обучения
                    Weights[j, i] += Corrections[j, 0, 0];//Обновление весов
                }
                Bias[i, 0, 0] += df * E;//Обновление смещений
            }
            DeltaList = DeltaNext;
        }
        protected override void RandomParams()//Генерация рандомных весов
        {
            for (int i = 0; i < Weights.n; i++)
            {
                for (int j = 0; j < Weights.m; j++)
                {
                    Weights[i, j] = Converter.RandomValues();
                    Bias[j, 0, 0] = 0.001;//rnd.NextDouble();
                }
            }
        }
        public override void Initialization()
        {
            //Neurons = new Tensor(Input, 1, 1);
            Corrections = new Tensor(Input, 1, 1);//Кол-во корректировок
            Weights = new Matrix(Input, Next);//Кол-во весов
            Output = new Tensor(Next, 1, 1);//Кол-во выходных значений
            Bias = new Tensor(Next, 1, 1);//Кол-во смещений
            RandomParams();
        }
    }
}
