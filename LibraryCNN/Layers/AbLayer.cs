using System;
namespace LibraryCNN
{
    [Serializable]
    public abstract class AbLayer
    {
        [field: NonSerialized]
        public event EventHandler<EventChangedParams> ParamsChanged;
        protected internal Tensor DeltaList { get; set; }// Дельты слоя
        protected internal Tensor Output { get; set; }// Выходной тензор
        internal Tensor InputTenz { get; set; }// Входной тензор
        public virtual Tensor Forward(in Tensor input) { return null; }
        public virtual void BackPropagation(in Tensor dout, in int Right, in double E, in double A) { }
        //[method: NonSerialized]
        protected virtual void OnParamsChanged(EventChangedParams e) { ParamsChanged?.Invoke(this, e); }
        protected virtual void RandomParams() { }
        public virtual void ReSizer(int inputNew) { }
    }
}
