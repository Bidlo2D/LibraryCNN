using LibraryCNN.Other;
using System;
using System.Collections.Generic;

namespace LibraryCNN
{
    [Serializable]
    public class ConvalutionLayer : AbLayer
    {
        //protected Random rnd = new Random();//рандомные числа для ядер сверток
        protected TypeActivation activation = TypeActivation.ReLU;
        protected int stride = 1, matrix = 1, countCore = 1, ratioPadding = 0;
        protected bool InBool;
        public virtual TypeActivation Activation //тип активации
        {
            get { return activation; }
            set { activation = value; OnParamsChanged(new EventChangedParams()); }
        }
        public virtual int Stride //значение скольжение
        {
            get { return stride; }
            set { if (value >= 1) { stride = value; OnParamsChanged(new EventChangedParams()); } }
        }
        public virtual int Matrix //размер ядра свертки
        {
            get { return matrix; }
            set { if (value >= 1) { matrix = value; OnParamsChanged(new EventChangedParams()); } }
        }
        public virtual int CountCore //кол-во ядер
        {
            get { return countCore; }
            set { if (value >= 1) { countCore = value; OnParamsChanged(new EventChangedParams()); } }
        }
        public virtual int RatioPadding //коэффицент падинга
        {
            get { return ratioPadding; }
            set { if (value >= 0) { ratioPadding = value; OnParamsChanged(new EventChangedParams()); } }
        }
        public ConvalutionLayer() { ParamsChanged += ConvalutionLayer_ParamsChanged; }
        private void ConvalutionLayer_ParamsChanged(object sender, EventChangedParams e)
        {
            Network.InitializationConv();
            Network.InitializationFully();
        }
        protected int Channel { get; set; }//число каналов выходных данных
        protected Tensor[] Cores { get; set; }//Хранилище ядер свертки
        protected Tensor[] CDout { get; set; }//Хранилище градиента весов
        protected int CountM { get; set; }// подсчет размерности карт
        protected int CoreCountM { get; set; }// подсчет размерности карт
        public virtual void Initialization(Tensor input)
        {
            Channel = input.SizeZ;
            Cores = new Tensor[CountCore];
            CDout = new Tensor[CountCore];
            RandomParams();
            InBool = true;
            PaddingOrNo(input);
            InBool = false;
        }
        public override void BackPropagation(in Tensor dout, in int Right, in double E, in double A)
        {
            CountingGradients(dout);
            if (Network.layersConv.Layer[0] == this)
            { NextLayerCountingGradients(dout); }
            UpdateWeight(E);
        }
        private void CountingGradients(in Tensor dout)
        {
            // расчитываем градиенты весов фильтров и смещений
            for (int f = 0; f < CountCore; f++)
            {
                for (int y = 0; y < dout.SizeX; y++)
                {
                    for (int x = 0; x < dout.SizeY; x++)
                    {
                        double delta = dout[f, y, x]; // запоминаем значение градиента

                        for (int i = 0; i < Matrix; i++)
                        {
                            for (int j = 0; j < Matrix; j++)
                            {
                                int i0 = i + y - 0;
                                int j0 = j + x - 0;

                                // игнорируем выходящие за границы элементы
                                if (i0 < 0 || i0 >= InputTenz.SizeX || j0 < 0 || j0 >= InputTenz.SizeY)
                                    continue;

                                // наращиваем градиент фильтра
                                for (int c = 0; c < Channel; c++)
                                    CDout[f][c, i, j] = delta * InputTenz[c, i0, j0]; //* X(c, i0, j0);
                            }
                        }
                    }
                }
            }
            //return (height, width);
        }
        private void NextLayerCountingGradients(in Tensor dout)
        {
            int pad = Matrix - 1 - 0; // заменяем величину дополнения
            Tensor dX = new Tensor(Channel, InputTenz.SizeX, InputTenz.SizeY); // создаём тензор градиентов по входу

            // расчитываем значения градиента
            for (int y = 0; y < InputTenz.SizeX; y++)
            {
                for (int x = 0; x < InputTenz.SizeY; x++)
                {
                    for (int c = 0; c < Channel; c++)
                    {
                        double sum = 0; // сумма для градиента

                        // идём по всем весовым коэффициентам фильтров
                        for (int i = 0; i < Matrix; i++)
                        {
                            for (int j = 0; j < Matrix; j++)
                            {
                                int i0 = y + i - pad;
                                int j0 = x + j - pad;

                                // игнорируем выходящие за границы элементы
                                if (i0 < 0 || i0 >= dout.SizeX || j0 < 0 || j0 >= dout.SizeY)
                                    continue;

                                // суммируем по всем фильтрам
                                for (int f = 0; f < CountCore; f++)
                                    sum += Cores[f][c, Matrix - 1 - i, Matrix - 1 - j] * dout[f, i0, j0]; // добавляем произведение повёрнутых фильтров на дельты
                            }
                        }
                        dX[c, y, x] = sum; // записываем результат в тензор градиента
                    }
                }
            }
            DeltaList = dX;
        }
        public override Tensor Forward(in Tensor input)//Проход по слою
        {
            InputTenz = input;
            for (int c = 0; c < CountCore; c++)//Проход всеми ядрами по тензору
            {
                PassageOneStep(c);
            }
            return Output;
        }
        protected virtual void PassageOneStep(in int c)//Прямой проход по изображению ядром свертки(Maping)
        {
            //Convolution
            int shiftX = 0; int shiftY = 0;//stride - скольжение по изображению
            int plusY = Matrix; int plusX = Matrix;//размеры свертки
            int Copy = CountM - 2 * RatioPadding;//кол-во сверток в скольжение по изображению
            int passageways = (int)Math.Pow(CountM - 2 * RatioPadding, 2);//кол-во сверток
            int OxMaping = 0, OyMaping = 0;//индексы для output
            for (int j = 0, plus = Copy; j < passageways; j++)
            {
                if (j == plus) 
                {
                    shiftX += Stride;
                    shiftY = 0;
                    plusX += Stride;
                    plusY = Matrix;
                    OxMaping++;
                    OyMaping = 0;
                    plus += Copy;
                }
                Tensor Conv = new Tensor(Channel, Matrix, Matrix);
                for (int ch = 0; ch < Channel; ch++)//Проход по всем каналам
                {
                    for (int x = shiftX, Ix = 0; x < plusX; x++, Ix++)
                    {
                        for (int y = shiftY, Iy = 0; y < plusY; y++, Iy++)
                        {
                            if(ch >= InputTenz.SizeZ 
                              || x >= InputTenz.SizeX
                              || y >= InputTenz.SizeY) 
                            { continue; }
                            Conv[ch, Ix, Iy] = InputTenz[ch, x, y];
                        }
                    }
                }
                Maping(Conv, in c, OxMaping + RatioPadding, OyMaping + RatioPadding);//3x3(5х5 и т.д) матрица части изображения
                shiftY += Stride; plusY += Stride; OyMaping++;
            }
        }
        protected virtual void PaddingOrNo(in Tensor input)
        {
            switch (RatioPadding)
            {
                case 0:
                    SizeNoPadding(input);
                    break;
                default:
                    SizePadding(input);
                    break;
            }
            Output = new Tensor(CountCore, CountM, CountM);
        }
        protected virtual void SizePadding(in Tensor input)
        {
            //подсчет размерности нового inputPad
            int PadXY = input.SizeX + 2 * RatioPadding;
            Tensor padInput = new Tensor(input.SizeZ, PadXY, PadXY, input.Right);
            //подсчет размерности outmap
            Channel = padInput.SizeZ;
            CountM = ((PadXY - Matrix) / Stride) + 1;
            CoreCountM = (int)Math.Pow(CountM, 2);
            if (InBool) { Padding(input, padInput); }
        }
        protected virtual void Padding(Tensor input, Tensor padInput)
        {
            for (int z = 0; z < input.SizeZ; z++)
            {
                for (int x = 0; x < input.SizeX; x++)
                {
                    for (int y = 0; y < input.SizeY; y++)
                    {
                        padInput[z, x + RatioPadding, y + RatioPadding] = input[z, x, y];
                    }
                }
            }
            InputTenz = padInput;
        }
        protected virtual void SizeNoPadding(in Tensor input)
        {
            //подсчет размерности outmap
            InputTenz = input;
            Channel = input.SizeZ;
            CountM = ((input.SizeX + 2 * RatioPadding - Matrix) / Stride) + 1;
            CoreCountM = (int)Math.Pow(CountM, 2);
        }
        protected override void RandomParams()//Функция генерации рандомных ядер свертки(Tensor[])
        {
            for (int count = 0; count < CountCore; count++)
            {
                Tensor core = new Tensor(Channel, Matrix, Matrix);
                for (int z = 0; z < core.SizeZ; z++)
                {
                    for (int x = 0; x < core.SizeX; x++)
                    {
                        for (int y = 0; y < core.SizeY; y++)
                        {
                            core[z, x, y] = Converter.RandomValues();
                        }
                    }
                }
                CDout[count] = new Tensor(Channel, Matrix, Matrix);
                Cores[count] = core;
            }
        }
        public void CustomValueRandomParams(List<Tensor> cores)//Функция генерации рандомных ядер свертки(Tensor[])
        {
            Cores = cores.ToArray();
        }
        protected virtual void UpdateWeight(double learningRate)
        {
            for (int index = 0; index < CountCore; index++)
            {
                for (int i = 0; i < Matrix; i++)
                {
                    for (int j = 0; j < Matrix; j++)
                    {
                        for (int d = 0; d < Channel; d++)
                        {
                            Cores[index][d, i, j] -= learningRate * CDout[index][d, i, j]; // вычитаем градиент, умноженный на скорость обучения
                            CDout[index][d, i, j] = 0; // обнуляем градиент фильтра
                        }
                    }
                }
            }
        }
        private void Maping(Tensor mass, in int c, int Ox, int Oy)
        {
            double convNum = 0;
            for (int ch = 0; ch < Channel; ch++)
            {
                for (int x = 0; x < mass.SizeX; x++)
                {
                    for (int y = 0; y < mass.SizeY; y++)
                    {
                        convNum += mass[ch, x, y] * Cores[c][ch, x, y];
                    }
                }
            }
            Output[c, Ox, Oy] = FuncActivations.Activation(Activation, convNum);
        }
    }
}
