using System;
using System.ComponentModel;

namespace LibraryCNN
{
    [Serializable]
    class ConvLayers : AbstractLayers<ConvalutionLayer>
    {
        public override Tensor DirectPassage(Tensor input)
        {
            if (Layer.Count != 0)
            {
                for (int i = 0; i < Layer.Count; i++)
                {
                    Layer[i].Forward(input);
                    input = Layer[i].Output;
                }
                return Layer[^1].Output;
            }
            else { return input; }
        }
        public override Tensor BackPassage(Tensor input, double LRate = 0.15, double A = 0.3)
        {
            for (int i = Layer.Count - 1; i >= 0; i--)
            {
                Layer[i].BackPropagation(input, input.Right, LRate, A);
                input = Layer[i].DeltaList;
            }
            return null;
        }
        protected override void Layer_ListChanged(object sender, ListChangedEventArgs e) { Network.InitializationConv(); Network.InitializationFully(); }
    }
}
