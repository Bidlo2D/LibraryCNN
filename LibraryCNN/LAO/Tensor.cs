using LibraryCNN.Other;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
namespace LibraryCNN
{
    [Serializable]
    public class Tensor : IComparable<Tensor>
    {
        private Neuron[] map;
        private int sizeZ, sizeX, sizeY, right;
        public string DirPath
        {
            get { return Path != null ? PathDir() : null; }
        }
        public int SizeZ
        {
            get { return sizeZ; }
            set { if (value >= 0) { sizeZ = value; } }
        }
        public int SizeX
        {
            get { return sizeX; }
            set { if (value >= 0) { sizeX = value; } }
        }
        public int SizeY
        {
            get { return sizeY; }
            set { if (value >= 0) { sizeY = value; } }
        }
        public int Right 
        { 
            get { return right; } 
            set { if (value >= 0) { right = value; } }
        }
        public string Path { get; set; }
        public int DW { get { return SizeZ * SizeX; } }
        public int FullSize { get { return SizeX * SizeY * SizeZ; } }
        public Tensor(double[,,] data, int right = 0)//Создание Tensor по размеру size
        {
            int n = 0;
            SizeZ = data.GetLength(0);
            SizeX = data.GetLength(1);
            SizeY = data.GetLength(2);
            Right = right;
            map = new Neuron[SizeX * SizeY * SizeZ];
            for (int z = 0; z < SizeZ; z++)
            {
                for (int x = 0; x < SizeX; x++)
                {
                    for (int y = 0; y < SizeY; y++)
                    {
                        map[n] = new Neuron(data[z, x, y]); n++;
                        //this[z, x, y] = data[z, x, y]; n++;
                    }
                }
            }
            //for (int n = 0; n < map.Length; n++) { map[n] = new Neuron(); }
        }
        public Tensor(int SizeZ, int SizeX, int SizeY, int Right = 0)//Создание Tensor по размеру size
        {
            this.SizeZ = SizeZ;
            this.SizeX = SizeX;
            this.SizeY = SizeY;
            this.Right = Right;
            map = new Neuron[SizeX * SizeY * SizeZ];
            for(int n = 0; n < map.Length; n++) { map[n] = new Neuron(); }
        }
        public Tensor(Bitmap image, TypeChannel channel, int Right, string Path) : this((int)channel, image.Width, image.Height, Right)//Интерпретация  данных из изображения в Tensor
        {
            this.Path = Path;
            for (int x = 0; x < image.Height; x++)//Проход по пикселям изображения 
            {
                for (int y = 0; y < image.Width; y++)
                {
                    Color cl = image.GetPixel(y, x);//получение пикселя изображение по его координатам
                    switch (channel)
                    {
                        case TypeChannel.RGB:
                            this[0, x, y] = (double)cl.R / 255;
                            this[1, x, y] = (double)cl.G / 255;
                            this[2, x, y] = (double)cl.B / 255;
                            break;
                        case TypeChannel.BW:
                            this[0, x, y] = (double)cl.R / 255; //(double)cl.R / 255;
                            break;
                        case TypeChannel.RGBA:
                            this[0, x, y] = (double)cl.R / 255;
                            this[1, x, y] = (double)cl.G / 255;
                            this[2, x, y] = (double)cl.B / 255;
                            this[3, x, y] = (double)cl.A / 255;
                            break;
                    }
                }
            }

        }
        //public Block(int z, int x, int y) { this[z,x,y].block = false; }
        public int Max() { return map.ToList().IndexOf(map.Max()); }
        public double Percent() { return map.ToList().Max(x => x.Value); }
        public double this[int z, int x, int y]//Индексация
        {
            get
            {
                return map[x * DW + y * SizeZ + z].Value;
            }
            set
            {
                map[x * DW + y * SizeZ + z].Value = value;
            }
        }
        public bool this[int z, int x, int y, double p]//Индексация
        {
            get
            {
                return map[x * DW + y * SizeZ + z].block;
            }
            set
            {
                if (Network.rnd.Next(0, 100) <= p){
                    map[x * DW + y * SizeZ + z].block = value;
                }
            }
        }
        public int CompareTo(Tensor Obj) { return FullSize.CompareTo(Obj.FullSize); }
        public List<Neuron> ToList() { return map.ToList(); }
        private string PathDir()
        {
            string result = "";
            string[] mass = Path.Split('\\');
            for (int i = 0; i < mass.Length - 1; i++)
            {
                if (i == mass.Length - 2) { result += $"{mass[i]}"; }
                else { result += $"{mass[i]}\\"; }
            }
            return result;
        }
    }
}
