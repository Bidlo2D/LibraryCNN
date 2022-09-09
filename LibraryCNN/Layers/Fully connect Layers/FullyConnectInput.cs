using LibraryCNN.Other;
using System;

namespace LibraryCNN
{
    [Serializable]
    public class FullyConnectInput : FullyConnectBias
    {
        public override TypeActivation TypeBackA //Тип производной для обратного прохода
        {
            get 
            { 
                if(Network.CountConv != 0)
                {
                    if (Network.layersConv.Layer[^1].GetType() == typeof(PoolingLayer)) { return TypeActivation.Input; }
                    else { return Network.layersConv.Layer[^1].Activation; }
                }
                else { return TypeActivation.Input; }
            }
        }
        public override int Input
        { 
            get 
            {
                //int save = input;
                if (Network.layersConv.Layer.Count != 0
                 && Network.layersConv.Layer[^1].Output != null)
                { return Network.layersConv.Layer[^1].Output.FullSize; }
                else if (Network.SelectionData != null 
                    && Network.SelectionData.Batches.Count != 0) 
                { return Network.SelectionData.Batches[0][0].FullSize; }
                else { return input; }
            }
        }
        public FullyConnectInput() { input = 0; }
        public override void BackPropagation(in Tensor delta, in int Right, in double E, in double A)//int Right, ForwardLayer PreDelta, double E, double A)
        {
            double dout = 0.0;
            bool nextB = Network.CountConv != 0 ? true : false; 
            Tensor DeltaNext = new Tensor(Neurons.SizeZ, Neurons.SizeX, Neurons.SizeY);
            for (int i = 0; i < Weights.m; i++)
            {
                double df = 0;
                for (int j = 0; j < Weights.n; j++)
                {
                    dout = FuncActivations.Deactivation(TypeBackA, Neurons, Neurons[j, 0, 0]);
                    if (nextB)
                    {
                        DeltaNext[j, 0, 0] += dout * (Weights[j, i] * delta[i, 0, 0]);//Градиент для следующего слоя
                    }
                    df += dout * delta[i, 0, 0];//градиент для смещений
                    double GRADw = Neurons[j, 0, 0] * delta[i, 0, 0];//Градиент данного слоя
                    Corrections[j, 0, 0] = E * GRADw + A * Corrections[j, 0, 0]; //Посчет обновления весов по градиенту и коф.обучения
                    Weights[j, i] += Corrections[j, 0, 0];//Обновление весовs
                }
                Bias[i, 0, 0] += df * E;//Обновление смещений
            }
            DeltaList = DeltaNext;
        }
    }
}
