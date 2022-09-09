using System;
using LibraryCNN.Other;
namespace LibraryCNN
{
    [Serializable]
    public class FullyConnectClassifier : FullyConnectLayer
    {
        private double[] Error { get; set; }
        //private int rndStochastic = 0;
        public override int Input
        {
            get
            {
                if (Network.SelectionData != null 
                    && Network.SelectionData.Batches.Count != 0) 
                { return Network.Max; }
                else { return input; }
            }
        }
        private int RndStochastic { get { return Network.rnd.Next(0, Input); } }
        public FullyConnectClassifier() { Error = new double[Input]; }
        public override Tensor Forward(in Tensor input)
        {
            Neurons = input;
            return null;
        }
        public override void BackPropagation(in Tensor delta, in int Right, in double E, in double A)
        {
            Loss = 0;
            for (int i = 0; i < Neurons.SizeZ; i++)
            {
                //double plus = Neurons[i, 0, 0] - Neurons[Right, 0, 0] + 1;
                if (i == Right)
                {
                    Error[i] = 1 - Neurons[i, 0, 0];//Подсчет ошибки(Положительный)
                    DeltaList[i, 0, 0] = Error[i] * FuncActivations.Deactivation(TypeBackA, Neurons, Neurons[i, 0, 0]);//Ошибка минус(+) Производная функции активации(результата)
                }
                else
                {
                    Error[i] = 0 - Neurons[i, 0, 0];//Подсчет ошибки(Отрицательный)
                    //Loss += plus >= 0 ? plus : 0;
                    DeltaList[i, 0, 0] = Error[i] * FuncActivations.Deactivation(TypeBackA, Neurons, Neurons[i, 0, 0]);//Ошибка минус(-) Производная функции активации(результата)
                }
                Loss += Math.Pow(Error[i], 2);
            }
            //Loss = Math.Pow(Error[RndStochastic], 2);// - (0.1 / Math.Pow(Network.NowEpoth, 0.3));
            Loss /= Neurons.SizeZ;
        }
        public override void Initialization() { DeltaList = new Tensor(Input, 1, 1); Error = new double[Input]; }
    }
}
