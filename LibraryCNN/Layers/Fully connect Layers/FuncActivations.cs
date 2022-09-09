using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibraryCNN
{
    static class FuncActivations
    {
        private const double alpha = 0.13;
        private static Tensor Output, Neurons;
        public static double Activation(TypeActivation forward, double x, Tensor input = null, Tensor output = null, int i = 0)//Активация нейрона
        {
            Output = output;
            Neurons = input;
            double result = 0;
            switch (forward)
            {
                case TypeActivation.Input:
                    result = x;
                    break;
                case TypeActivation.ReLU:
                    result = ReLU(x);
                    break;
                case TypeActivation.Tangent:
                    result = Tangent(x);
                    break;
                case TypeActivation.Sigmoid:
                    result = Sigmoid(x);
                    break;
                case TypeActivation.Softmax:
                    result = SoftMax(x, i);
                    break;
                case TypeActivation.ELU:
                    result = ELU(x);
                    break;
            }
            return result;
        }
        private static double ExpTang(double x)
        {
            double expF = Math.Exp(x);
            if (Double.IsInfinity(expF)) { return -1; }
            else { return expF; }
        }
        private static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        private static double Tangent(double x)
        {
            return 2 / (1 + Math.Exp(x)) - 1;
        }
        public static double ReLU(double x)
        {
            if (x <= 0) { return 0; }
            else { return x; }
        }
        private static double ELU(double x)
        {
            if (x > 0) { return x; }
            else { return alpha * (Math.Exp(x) - 1); }
        }
        private static double SoftMax(double x, int index)
        {
            try
            {
                if (index < Output.SizeZ - 1)
                {
                    return Math.Exp(x);
                }
                else
                {
                    for (int i = 0; i < Output.SizeZ; i++)
                    {
                        Output[i, 0, 0] = Output[i, 0, 0] / SumSoftmaxActive();
                    }
                    return Math.Exp(x) / SumSoftmaxActive();
                }
            }
            catch (Exception) { return x; }
        }
        private static double SumSoftmaxActive()
        {
            double sum = 0;
            for (int i = 0; i < Output.SizeZ; i++)
            {
                sum += Output[i, 0, 0];
            }
            return sum;
        }
        //Производные функций активаций
        public static double Deactivation(TypeActivation back, Tensor input, double x)//Производная
        {
            Neurons = input;
            double result = 0;
            switch (back)
            {
                case TypeActivation.Input:
                    result = x;
                    break;
                case TypeActivation.ReLU:
                    result = DerivativeReLU(x);
                    break;
                case TypeActivation.Tangent:
                    result = DerivativeTangent(x);
                    break;
                case TypeActivation.Sigmoid:
                    result = DerivativeSigmoid(x);
                    break;
                case TypeActivation.Softmax:
                    result = DerivativeSoftMax(x);
                    break;
                case TypeActivation.ELU:
                    result = DerivativeELU(x);
                    break;
            }
            return result;
        }
        private static double DerivativeSigmoid(double x)
        {
            return x * (1 - x);
        }
        private static double DerivativeTangent(double x)
        {
            return 1 - Math.Pow(x, 2);
        }
        private static double DerivativeELU(double x)
        {
            if (x > 0) { return 1; }
            else { return x + alpha; }
        }
        private static double DerivativeReLU(double x)
        {
            if (x > 0) { return 1; }
            else { return 0; }
        }
        private static double DerivativeSoftMax(double x)
        {
            try
            {
                return (x * SumSoftmaxDeactive(1)) / Math.Pow(SumSoftmaxDeactive(0), 2);
            }
            catch (Exception) { return x; }
        }
        private static double SumSoftmaxDeactive(int iMode)
        {
            double sum = 0;
            for (int i = iMode; i < Neurons.SizeZ; i++)
            {
                sum += Neurons[i, 0, 0];
            }
            return sum;
        }
    }
}
