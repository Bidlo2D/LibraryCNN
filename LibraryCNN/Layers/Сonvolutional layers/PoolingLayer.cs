using System;
using LibraryCNN.Other;
namespace LibraryCNN
{
    [Serializable]
    public class PoolingLayer : ConvalutionLayer
    {
        private int scale = 2;
        private TypePool mode = TypePool.Max;
        public override int Stride { set { } } 
        public override int Matrix { set { } } 
        public override int CountCore { set { } }
        public override TypeActivation Activation { set { } }
        //public override TypeActivation Activation { get; }
        public int Scale { get { return scale; } 
            set { if (value > 0) { scale = value; OnParamsChanged(new EventChangedParams()); } } }//кофф уменьшения изображения
        public TypePool Mode { get { return mode;  } 
            set { mode = value; OnParamsChanged(new EventChangedParams()); } }//режим пуллинга
        int matrixSize { get { return Scale * Scale; } }//размер матрицы пуллинга
        Tensor mask { get; set; }//маска для обратного прохода(MaxPool)
        private double pool;
        private int countDout;
        private int Ix, Iy;
        public override Tensor Forward(in Tensor input)
        {
            //PaddingOrNo(input);
            // проходимся по каждому из каналов
            for (int d = 0; d < input.SizeZ; d++)
            {
                for (int i = 0; i < input.SizeX; i += Scale)
                {
                    for (int j = 0; j < input.SizeY; j += Scale)
                    {
                        if(d >= Output.SizeZ 
                        || i >= Output.SizeX
                        || j >= Output.SizeY
                        || Ix >= mask.SizeX
                        || Iy >= mask.SizeY)
                        { continue; }
                        int aI = 0;
                        pool = 0;
                        for (int y = i; y < i + Scale; y++)
                        {
                            for (int x = j; x < j + Scale; x++)
                            {
                                if (d >= input.SizeZ
                                 || x >= input.SizeX
                                 || y >= input.SizeY)
                                { continue; }
                                aI++;
                                pool = Pooling(input[d, x, y], aI, x, y);
                            }
                        }
                        //Pooling(values, (int)mode, d, i / Scale, j / Scale);
                        Output[d, i / Scale + RatioPadding, j / Scale] = pool; // записываем в выходной тензор найденный максимум
                        mask[d, Ix, Iy] = 1;
                    }
                }
            }
            return Output;
        }
        protected override void PaddingOrNo(in Tensor input)
        {
            switch (RatioPadding)
            {
                case 0:
                    InputTenz = input;
                    break;
                default:
                    SizePadding(input);
                    break;
            }
            Output = new Tensor(InputTenz.SizeZ, InputTenz.SizeX / Scale, InputTenz.SizeY / Scale);
            mask = new Tensor(InputTenz.SizeZ, InputTenz.SizeX, InputTenz.SizeY);
        }
        protected override void SizePadding(in Tensor input)
        {
            //подсчет размерности нового inputPad
            int PadXY = input.SizeX + 2 * RatioPadding;
            Tensor padInput = new Tensor(input.SizeZ, PadXY, PadXY, input.Right);
            if (InBool) { Padding(input, padInput); }
        }
        //TODO: fixed out to index - dout
        public override void BackPropagation(in Tensor dout, in int Right, in double E, in double A)
        {
            countDout = dout.FullSize;
            DeltaList = new Tensor(mask.SizeZ, mask.SizeX, mask.SizeY);
            for (int d = 0; d < mask.SizeZ; d++)
                for (int i = 0; i < mask.SizeX; i++)
                    for (int j = 0; j < mask.SizeY; j++)
                    {
                        if (d >= dout.SizeZ
                          || i >= dout.SizeX
                          || j >= dout.SizeY)
                        { continue; }
                        DeltaList[d, i, j] = BackPropagation(dout[d, i / Scale, j / Scale], mask[d, i, j]); // умножаем градиенты на маску
                    }
        }
        private double BackPropagation(double input, double mask)
        {
            double result = 0;
            switch (Mode)
            {
                case TypePool.Max:
                    result = BackMaxPooling(input, mask);
                    break;
                case TypePool.Avg:
                    result = input / countDout;
                    break;
                case TypePool.Sum:
                    result = input;
                    break;
            }
            return result;
        }
        private double BackMaxPooling( double input,  double mask)
        {
            return input * mask;
        }
        private double Pooling(double input, int index, int x, int y)
        {
            double result = 0;
            switch ((int)Mode)
            {
                case 0:
                    result = MaxPooling(input, x, y);
                    break;
                case 1:
                    result = AveragePooling(input, index);
                    break;
                case 2:
                    result = SumPooling(input);
                    break;
            }
            return result;
        }
        private double MaxPooling(double input, int x, int y)
        {
            if (pool < input)
            {
                Ix = x; Iy = y;
                pool = input;
            }
            return pool;
        }
        private double AveragePooling(double input, int index)
        {
            pool += input;
            if (index < matrixSize) { return pool; }
            else { return pool / matrixSize; }
        }
        private double SumPooling(double input)
        {
            return pool += input;
        }
        public override void Initialization(Tensor input) 
        {
            InBool = true;
            PaddingOrNo(input);
            InBool = false;
        }
    }
}
