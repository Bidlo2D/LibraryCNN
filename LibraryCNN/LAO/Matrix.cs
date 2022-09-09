using System;
namespace LibraryCNN
{
    [Serializable]
    public class Matrix
    {
        double[,] v;

        public int n; public int m;
        public Matrix(int sizeX, int sizeY)
        {
            n = sizeX; m = sizeY;
            v = new double[sizeX, sizeY];
        }
        public Matrix(double[,] mass)
        {
            n = mass.GetLength(0); m = mass.GetLength(1);
            v = mass;
        }
        public double this[int indexX, int indexY]
        {
            get
            {
                return v[indexX, indexY];
            }
            set
            {
                v[indexX, indexY] = value;
            }
        }
    }
}
