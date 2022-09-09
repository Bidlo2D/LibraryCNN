using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text.Json;

namespace LibraryCNN.Other
{
    public static class Converter
    {
        private static Random rnd = new Random();
        public static bool IsLoad { get; set; }
        public static Tensor RandomImage(int Z, int X, int Y, int right = 0)
        {//new int[rightMax - rightMin];
            Tensor image = new Tensor(Z, X, Y, right);
            for (int z = 0; z < image.SizeZ; z++)
            {
                for (int x = 0; x < image.SizeX; x++)
                {
                    for (int y = 0; y < image.SizeY; y++)
                    {
                        image[z, x, y] = RandomValues(-10000, 10000);
                    }
                }
            }
            return image;
        }
        public static Tensor RandomImage(int Z, int X, int Y, out Tensor image, int rightMin = 0, int rightMax = 10)
        {
            int right = rnd.Next(rightMin, rightMax);
            image = new Tensor(Z, X, Y, right);
            for (int z = 0; z < image.SizeZ; z++)
            {
                for (int x = 0; x < image.SizeX; x++)
                {
                    for (int y = 0; y < image.SizeY; y++)
                    {
                        image[z, x, y] = RandomValues(-10000, 10000);
                    }
                }
            }
            return image;
        }
        public static double RandomValues(int min = -1000, int max = 1000)
        {
            double x;
            int PlusOrMinus = rnd.Next(0, 2);
            if (PlusOrMinus > 0) { x = (rnd.Next(min, max) / 10000f) * (-1); }
            else { x = rnd.Next(min, max) / 10000f; }
            return x;
        }
        public static void SaveBatch(Stream file, Batch batch)
        {
            var binFormatter = new BinaryFormatter();
            binFormatter.Serialize(file, batch);
        }
        public static void SaveBatch(string path, Batch batch)
        {
            using (FileStream file = new FileStream(path, FileMode.OpenOrCreate))
            {
                var binFormatter = new BinaryFormatter();
                binFormatter.Serialize(file, batch);
            }
        }
        public static Batch LoadBatch(Stream file)
        {
            var binFormatter = new BinaryFormatter();
            Batch resultLoad = binFormatter.Deserialize(file) as Batch;
            return resultLoad;
        }
        public static Batch LoadBatch(string path)
        {
            Batch resultLoad;
            using (FileStream file = new FileStream(path, FileMode.OpenOrCreate))
            {
                var binFormatter = new BinaryFormatter();
                resultLoad = binFormatter.Deserialize(file) as Batch;
            }
            return resultLoad;
        }
        public static AbstractLayers<AbLayer> LoadNet(Stream file)
        {
            Network.layersConv.Layer.Clear();
            Network.layersFully.Layer.Clear();
            var binFormatter = new BinaryFormatter();
            AbstractLayers<AbLayer> resultLoad = binFormatter.Deserialize(file) as AbstractLayers<AbLayer>;
            IsLoad = true;
            return resultLoad;
        }
        public static void SaveNet(Stream file)
        {
            var binFormatter = new BinaryFormatter();
            binFormatter.Serialize(file, Network.NET);
        }
    }
}
