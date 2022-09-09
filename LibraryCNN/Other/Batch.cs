using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibraryCNN.Other
{
    [Serializable]
    public class Batch
    {
        //NonSerialized
        [NonSerialized]
        private List<FileInfo> Data = new List<FileInfo>();
        //Serializable
        public List<Tensor> dataSet = new List<Tensor>();
        public List<Tensor[]> Batches = new List<Tensor[]>();
        public List<Tensor> Test = new List<Tensor>();
        private (int,int) MaxSizeHW { get; set; }
        private string DirPath { get; set; }
        private int _batchSize, _percent;
        public int PatchSize { get; set; }
        public int Percent 
        {
            get { return _percent; }
            set { if (value > 0) { _percent = value; ChangePercent();  } }
        }
        public int BatchSize
        {
            get { return _batchSize; }
            set { if (value > 0) { _batchSize = value; ChangeSizeBatch(); } }
        }
        public Batch() { }
        public Batch(string dirPath, int batchSize, int percent)
        {
            _batchSize = batchSize;
            _percent = percent;
            DirPath = dirPath;
            DirectoryInfo dir = new DirectoryInfo(dirPath);
            FileInfo[] DataLocal = dir.GetFiles().Where(file => file.Extension == ".png" || file.Extension == ".jpg").ToArray();
            MaxSizeHW = Max(DataLocal);
            //1 - Interpretation image to tensor
            int testCount = (int)(Data.Count / 100f * percent);
            int randomIndexTest = Network.rnd.Next(0, Data.Count - testCount);
            for (int i = 0; i < Data.Count; i++){
                string PathImage = dirPath + $@"\{Data[i].Name}";
                dataSet.Add(Answer(Data[i], PathImage));
            }
            TestData(randomIndexTest, testCount);
            TrainData();
        }
        private void ChangeSizeBatch()
        {
            if (dataSet != null)
            {
                TrainData();
            }
        }
        private void ChangePercent()
        {
            if (dataSet != null 
                && Batches != null 
                && Test != null)
            {
                int testCount = (int)(dataSet.Count / 100f * _percent);
                int randomIndexTest = Network.rnd.Next(0, dataSet.Count - testCount);
                TestData(randomIndexTest, testCount);
                TrainData();
            }
        }

        private void TestData(int randomIndexTest, int testCount)
        {
            Test.Clear();
            for (int i = randomIndexTest; i < randomIndexTest + testCount; i++)
            {
                Test.Add(dataSet[i]);
            }
        }
        private void TrainData()
        {
            Batches.Clear();
            //List<Tensor> newDataSet = DeleteTestOfDataSet();
            for (int i = 0; i < dataSet.Count;)
            {
                List<Tensor> miniBatch = new List<Tensor>();
                for (; miniBatch.Count < _batchSize;)
                {
                    if(i >= dataSet.Count) { break; }
                    if (!Test.Contains(dataSet[i]))
                    {
                        miniBatch.Add(dataSet[i]);
                    }
                    i++;
                }
                if (miniBatch.Count != 0) { Batches.Add(miniBatch.ToArray()); }
            }
        }
        private List<Tensor> DeleteTestOfDataSet()
        {
            List<Tensor> dataNew = new List<Tensor>();
            foreach (var t in dataSet) 
            {
                if (!Test.Contains(t))
                {
                    dataNew.Add(t);
                }
            }
            return dataNew;
        }
        private Tensor Answer(FileInfo data, string Path)
        {
            // TODO: Fix zero answer
            string strChar = "";
            Bitmap image = new Bitmap(Path);
            string[] name = data.Name.Split(new char[] { '.' });
            if(name[0].Length > 2) { strChar = name[0][^2].ToString() + name[0][^1].ToString(); }
            Int32.TryParse(strChar, out int answer);
            if(image.Height != MaxSizeHW.Item1
                || image.Width != MaxSizeHW.Item2)
            { image = new Bitmap(image, new Size(MaxSizeHW.Item1, MaxSizeHW.Item2)); }
            Tensor Output = new Tensor(image, Network.Channel, answer, Path);
            return Output;
        }
        private Bitmap ImageCorection(Bitmap bitmap)
        {
            if (bitmap.Height > bitmap.Width)
            { return new Bitmap(bitmap, new Size(bitmap.Height, bitmap.Height)); }
            else if(bitmap.Height == bitmap.Width) 
            { return bitmap; }
            else
            { return new Bitmap(bitmap, new Size(bitmap.Width, bitmap.Width)); }
        }
        private (int,int) Max(FileInfo[] data) 
        {
            List<Tensor> dataBitmap = new List<Tensor>();
            for (int i = 0; i < data.Length; i++)
            {
                string PathImage = DirPath + $@"\{data[i].Name}";
                Bitmap bitmap = new Bitmap(PathImage);
                if(bitmap.Width > 500 || bitmap.Height > 500) { continue; }
                else 
                {
                    bitmap = ImageCorection(bitmap);
                    dataBitmap.Add(new Tensor(bitmap, Network.Channel, 0, ""));
                    Data.Add(data[i]);
                }
            }
            Tensor maxTensor = dataBitmap.Max();//Max(item => item.Width * item.Height * (int)Channel)
            return (maxTensor.SizeX, maxTensor.SizeY);
        }
    }
}
