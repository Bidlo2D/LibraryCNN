using LibraryCNN.Other;
using System;
using System.Linq;

namespace LibraryCNN
{
    [Serializable]
    public class ConvalutionLayerBias : ConvalutionLayer
    {
        double[] db;//Градиенты смещений
        double[] Bias;//Смещения
        protected override void PassageOneStep(in int c)//Прямой проход по изображению ядром свертки(BiasMaping)
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
                            if (ch >= InputTenz.SizeZ
                              || x >= InputTenz.SizeX
                              || y >= InputTenz.SizeY)
                            { continue; }
                            Conv[ch, Ix, Iy] = InputTenz[ch, x, y];
                        }
                    }
                }
                BiasMaping(Conv, c, OxMaping + RatioPadding, OyMaping + RatioPadding);//3x3(5х5 и т.д) матрица части изображения
                shiftY += Stride; plusY += Stride; OyMaping++;
            }
        }
        void BiasMaping(in Tensor mass, in int c, in int Ox, in int Oy)
        {
            double convNum = Bias[c];
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
            Output[c, Ox, Oy] = convNum;//ReLU(convNum);
        }
        public override void BackPropagation(in Tensor dout, in int Right, in double E, in double A)
        {
            CountingGradients(dout);
            if (Network.layersConv.Layer[0] != this)
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
                                    CDout[f][c, i, j] += delta * InputTenz[c, i0, j0]; //* X(c, i0, j0);
                            }
                        }
                        db[f] += delta; // наращиваем градиент смещения
                    }
                }
            }
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
                                {
                                    sum += Cores[f][c, Matrix - 1 - i, Matrix - 1 - j] * dout[f, i0, j0];
                                    //sum += coresReverse[f][c, i, j] * dout[f, i0, j0];
                                } // добавляем произведение повёрнутых фильтров на дельты
                            }
                        }
                        dX[c, y, x] = sum; // записываем результат в тензор градиента
                    }
                }
            }
            DeltaList = dX;
        }
        protected override void UpdateWeight(double learningRate)
        {
            for (int c = 0; c < CountCore; c++)
            {
                for (int i = 0; i < Matrix; i++)
                {
                    for (int j = 0; j < Matrix; j++)
                    {
                        for (int d = 0; d < Channel; d++)
                        {
                            Cores[c][d, i, j] -= learningRate * CDout[c][d, i, j]; // вычитаем градиент, умноженный на скорость обучения
                            CDout[c][d, i, j] = 0; // обнуляем градиент фильтра
                        }
                    }
                }
                Bias[c] += learningRate * db[c]; // вычитаем градиент, умноженный на скорость обучения
                db[c] = 0; // обнуляем градиент веса смещения
            }
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
/*                            int PlusOrMinus = Network.rnd.Next(0, 2);
                            if (PlusOrMinus > 0) { core[z, x, y] = Network.rnd.NextDouble() * (-1); }
                            else { core[z, x, y] = Network.rnd.NextDouble(); }*/
                        }
                    }
                }
                Bias[count] = 0.001;//rnd.NextDouble();
                CDout[count] = new Tensor(Channel, Matrix, Matrix);
                Cores[count] = core;
            }
        }
        public override void Initialization(Tensor input)
        {
            Channel = input.SizeZ;
            db = new double[CountCore];
            Bias = new double[CountCore];
            Cores = new Tensor[CountCore];
            CDout = new Tensor[CountCore];
            RandomParams();
            InBool = true;
            PaddingOrNo(input);
            InBool = false;
        }
    }
}
