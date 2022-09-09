using LibraryCNN.Other;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace LibraryCNN
{
    public class Program
    {
        static ConvalutionLayerBias conv = new ConvalutionLayerBias();
        static PoolingLayer pool = new PoolingLayer();
        static ConvalutionLayerBias conv1 = new ConvalutionLayerBias();
        static PoolingLayer pool1 = new PoolingLayer();
        static FullyConnectInput start = new FullyConnectInput();
        static FullyConnectBias hidden1 = new FullyConnectBias();
        static FullyConnectBias hidden2 = new FullyConnectBias();
        static FullyConnectClassifier end = new FullyConnectClassifier();
        static bool Stop { get; set; } = false;
        class Person
        {
            public string Name { get; set; }
            public int Age { get; set; }

        }
        static void Main(string[] args)
        {
            Network.Epoth = 5000;
            Network.LRate = 0.095;
            Network.Channel = TypeChannel.BW;
            //Continue train network
            LoadBatch("32x32BatchNumbers");
            CreateCNN();
            //TestConvInput("convBatch1");
            //CreateCNN();
            StartLearning();
            //SaveBatch("testBatch1");
            SaveNet("testNet1");
        }
        private static void SaveNet(string nameNet)
        {
            using (FileStream fs = new FileStream(@$"C:\Games\Programs\Эволюция\CNN\SaveNetTest\ConsoleNET\{nameNet}.bin", FileMode.OpenOrCreate))
            {
                Converter.SaveNet(fs);
            }
        }
        private static void SaveBatch(string nameBatch)
        {
            using (FileStream fs = new FileStream(@$"C:\Games\Programs\Эволюция\CNN\SaveNetTest\ConsoleNET\{nameBatch}.bin", FileMode.OpenOrCreate))
            {
                Converter.SaveBatch(fs, Network.SelectionData);
            }
        }
        private static void LoadNet(string nameNet = "NetConsole")
        {
            using (FileStream fs = new FileStream(@$"C:\Games\Programs\Эволюция\CNN\SaveNetTest\ConsoleNET\{nameNet}.bin", FileMode.OpenOrCreate))
            {
                Network.NET = Converter.LoadNet(fs);
            }
        }
        private static void LoadBatch(string nameBatch = "batchConsole")
        {
            using (FileStream fs = new FileStream(@$"C:\Games\Programs\Эволюция\CNN\SaveNetTest\ConsoleNET\{nameBatch}.bin", FileMode.OpenOrCreate))
            {
                Network.SelectionData = Converter.LoadBatch(fs);
            }
        }
        private static void CreateCNN()
        {
            //hidden
            hidden1.Input = 10;
            hidden1.TypeBackA = TypeActivation.Sigmoid;
            hidden2.Input = 15;
            hidden2.TypeBackA = TypeActivation.Sigmoid;
            end.TypeBackA = TypeActivation.Softmax;
            //conv
            conv.Stride = 1;
            conv.Matrix = 5;
            conv.CountCore = 3;
            //conv.RatioPadding = 2;
            conv.Activation = TypeActivation.Tangent;
            conv1.Stride = 1;
            conv1.Matrix = 5;
            conv1.CountCore = 16;
            conv1.Activation = TypeActivation.Tangent;
            pool.Mode = TypePool.Max;
            pool1.Mode = TypePool.Max;
            //TODO: Problems in convolution layers
            Network.Add(conv);
            Network.Add(pool);
            //Network.Add(conv1);
            //Network.Add(pool1);
            Network.Add(start);
            Network.Add(hidden1);
            Network.Add(hidden2);
            Network.Add(end);
            //conv.CustomValueRandomParams(CoresGeneration());
        }
        private static void TestConvInput(string name)
        {
            List<Tensor[]> trainBatch = new List<Tensor[]>(); int t = 0;
            List<Tensor> testBatch = new List<Tensor>();
            foreach (var Batch in Network.SelectionData.Batches)
            {
                Tensor[] b = new Tensor[Batch.Length];
                foreach (var miniBatch in Batch)
                {
                    //b[t] = Network.ForwardNet(miniBatch);
                    b[t].Right = t;
                    t++;
                }
                trainBatch.Add(b);
            }
            int test = 0;
            foreach (var miniBatch in Network.SelectionData.Test)
            {
                //testBatch.Add(Network.ForwardNet(miniBatch));
                testBatch[test].Right = test;
                test++;
            }
            Network.SelectionData.dataSet = trainBatch[0].ToList();
            Network.SelectionData.Batches = trainBatch;
            Network.SelectionData.Test = testBatch;
            SaveBatch(name);
        }
        private static void StartLearning()
        {
            Stopwatch time = new Stopwatch();
            for (int e = 0; e < Network.Epoth; e++, Network.NowEpoth++)
            {
                time.Restart();
                int rights = 0;
                double loss = 0;
                foreach (var Batch in Network.SelectionData.Batches)
                {
                    foreach (var miniBatch in Batch)
                    {
                        Network.ForwardNet(miniBatch);
                        Network.BackNet(miniBatch);
                        loss += Network.Loss;
                        if (Network.Answer == miniBatch.Right)
                        { rights++; }
                    }
                }
                if (Stop) { break; }
                time.Stop();
                (double, int) resultTest = TestNet();
                double SecondEpoth = time.ElapsedMilliseconds / 1000.0;
                double RemainedMinute = (SecondEpoth / 60) * (Network.Epoth - (e + 1));
                double RemainedHours = (RemainedMinute / 60);
                loss /= Network.SelectionData.Batches.Count;
                Console.WriteLine($"Learning data - l = {loss, 10}, r = {rights}");
                Console.WriteLine($"epoth - {e + 1} - epoth second - {SecondEpoth}s - remained ~ {Math.Round(RemainedMinute,2)}m ~ {Math.Round(RemainedHours, 2)}h");
                Console.WriteLine($"Testing data - l = {resultTest.Item1}, r = {resultTest.Item2}");
                Console.WriteLine("");
                //Console.SetCursorPosition(0, 0);
            }
        }
        private static (double, int) TestNet()
        {
            int rights = 0;
            double loss = 0;
            foreach (var item in Network.SelectionData.Test)
            {
                Network.ForwardNet(item);
                Network.BackNoRefresh(item);
                loss += Network.Loss;
                if (Network.Answer == item.Right)
                { rights++; }
            }
            return (loss, rights);
        }
        public static Tensor[] RandomTensorMass(int count, int z = 1, int x = 27, int y = 27)
        {
            List<int> listRights = Enumerable.Range(0, count).ToList();
            Tensor[] a = new Tensor[count];
            for (int i = 0; i < a.Length;i++)
            {
                int randomRight = Network.rnd.Next(0, listRights.Count);
                a[i] = Converter.RandomImage(z, x, y, listRights[randomRight]);
                listRights.RemoveAt(randomRight);
            }
            return a;
        }
        private static List<Tensor> CoresGeneration()
        {
            //1
/*              3   1   2   1   1
                3   3   1   2   1
                1   1   2   2 - 1
              - 1  2   3   3 - 1
                1 - 2  2   2   0*/
            List<Tensor> cores = new List<Tensor>();
            double[,,] core1 =
            { 
                 { { 3,  1,  2,  1,  1 },
                 {   3,  3,  1,  2,  1 },
                 {   1,  1,  2,  2, -1 },
                 {  -1,  2,  3,  3, -1 },
                 {   1, -2,  2,  2,  0 } }
            };
            double[,,] core2 =
            {
                 { { -1, -1, -2, -1,  1  },
                 {    1,  3,  3,  0,  2  },
                 {   -1,  2,  1, -1,  2  },
                 {   -1,  2, -2,  2,  3  },
                 {    2,  1,  3,  0, -2  } }
            };
//2
/*             -1 - 1 - 2 - 1  1
                1   3   3   0   2
              - 1  2   1 - 1  2
              - 1  2 - 2  2   3
                2   1   3   0 - 2*/
            double[,,] core3 =
            {
                 { { 1, -2,  1, -1, -2 },
                 {  -2, -1,  2,  0,  2 },
                 {   3,  1,  3, -1,  2 },
                 {   0,  0,  0, -1,  1 },
                 {  -1, -1, -1,  2,  3 } }
            };
//3
/*              1 - 2  1 - 1 - 2
              - 2 - 1  2   0   2
                3   1   3 - 1  2
                0   0   0 - 1  1
              - 1 - 1 - 1  2   3*/
            Tensor core1Tensor = new Tensor(core1);
            Tensor core2Tensor = new Tensor(core2);
            Tensor core3Tensor = new Tensor(core3);
            cores.Add(core1Tensor);
            cores.Add(core2Tensor);
            cores.Add(core3Tensor);
            return cores;
        }
        private static List<Tensor> TrainGeneration()
        {
            double[,,] testTensor =
            {
                 { { 0, -2, 0,  2,  -2 },
                 { 3, -2, -2, 1,  -2 },
                 { 1, 2,  0,  3,  -1 },
                 { 0, 1,  1,  -2, -2 },
                 { 2, 3,  -1, -1, 3 } }
            };
            List<Tensor> train1 = new List<Tensor>();
            Tensor batch = new Tensor(testTensor);
            train1.Add(batch);
            return train1;
        }
        private static void GenerationBatchRandom()
        {
            var train1 = TrainGeneration();
            Network.SelectionData = new Batch();
            //var train1 = RandomTensorMass(6, 1, 32, 32);//train batch 1
            //var train2 = RandomTensorMass(6, 1, 32, 32);//train batch 2
            var test = RandomTensorMass(1, 1, 5, 5);//test
            Network.SelectionData.dataSet = train1.ToList();//Enumerable.Union(train1, train2).ToList();
            Network.SelectionData.Batches.Add(train1.ToArray());
            //Network.SelectionData.Batches.Add(train2);
            Network.SelectionData.Test = test.ToList();
            //Network.SelectionData = new Batch(@"C:\Games\Programs\Fonts\Numbers(32x32)", 10,35);
        }
    }
}
