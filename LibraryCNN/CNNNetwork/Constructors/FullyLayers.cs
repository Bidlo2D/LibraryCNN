using System;
using System.ComponentModel;
using System.Linq;

namespace LibraryCNN
{
    [Serializable]
    class FullyLayers : AbstractLayers<FullyConnectLayer>
    {
        public void Drop(double p)
        {
            for (int i = 0; i < Layer.Count - 2; i++)
            {
                Layer[i].DropOut(p);
            }
        }
        public void Unlock()
        {
            for (int i = 0; i < Layer.Count - 2; i++)
            {
                Layer[i].Unlocked(100);
            }
        }
        public override Tensor DirectPassage(Tensor input)
        {
            for (int i = 0; i < Layer.Count; i++)
            {
                Layer[i].Forward(input);
                input = Layer[i].Output;
            }
            return null;
        }
        public override Tensor BackPassage(Tensor input, double LRate = 0.15, double A = 0.3)
        {
            for (int i = Layer.Count - 1; i >= 0; i--)
            {
                Layer[i].BackPropagation(input, input.Right, LRate, A);
                input = Layer[i].DeltaList;
            }
            return Layer[0].DeltaList;
        }
        protected override void Layer_ListChanged(object sender, ListChangedEventArgs e) { Network.InitializationFully(); }
    }
}
