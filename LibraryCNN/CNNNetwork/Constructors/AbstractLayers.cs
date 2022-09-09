using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibraryCNN
{
    [Serializable]
    public class AbstractLayers<T> where T : AbLayer
    {
        public AbstractLayers() { Layer.ListChanged += Layer_ListChanged;  }
        public BindingList<T> Layer { get; set; } = new BindingList<T>();
        public void Load(T layer)
        {
            Layer.ListChanged -= Layer_ListChanged;
            Layer.Add(layer);
            Layer.ListChanged += Layer_ListChanged;
        }
        public virtual void Delete(int index) { Layer.RemoveAt(index); }
        public virtual Tensor DirectPassage(Tensor input) { return default; }
        public virtual Tensor BackPassage(Tensor input, double LRate = 0.15, double A = 0.3) { return default; }
        protected virtual void Layer_ListChanged(object sender, ListChangedEventArgs e) { }
    }
}
