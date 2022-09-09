using LibraryCNN.Other;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading;

namespace LibraryCNN
{
    [Serializable]
    public static class Network
    {
        public static Random rnd = new Random();
        private static double lRate = 0.15, aRatio = 0.3;
        private static double probability = 0;
        private static Batch selectionData;
        private static int commaConfidence = 2;
        private static int inputImageSize = 1;
        private static Tensor test = Converter.RandomImage(1, 32, 32);
        public static Batch SelectionData { 
            get { return selectionData; } 
            set 
            {
                selectionData = value;
                CheckLoad();
            }  
        }
        public static TypeChannel Channel { get; set; } = TypeChannel.RGB;
        internal static ConvLayers layersConv = new ConvLayers();
        internal static FullyLayers layersFully = new FullyLayers();
        private static int epoth = 0, nowEpoth = 1;
        public static AbstractLayers<AbLayer> NET { internal get { return UnionLayers(); } set { SeparationLayers(value); } }
        public static int InputImageSize
        {
            get
            {
                if (layersConv.Layer.Count != 0)
                {
                    if (layersConv.Layer[0].InputTenz != null)
                    {
                        inputImageSize = layersConv.Layer[0].InputTenz.SizeX;
                    }
                }
                return inputImageSize;
            }
        }
        public static int СommaConfidence 
        { 
            get { return commaConfidence; }
            set { if(value >= 0 && value <= 15) { commaConfidence = value; } }
        }
        public static int Epoth
        {
            get { return epoth; }
            set { if (value > 0) { epoth = value; } }
        }
        public static int NowEpoth
        {
            get { return nowEpoth; }
            set { if (value > 0) { nowEpoth = value; } }
        }
        public static double LRate
        {
            get { return lRate; }
            set { if (value > 0) { lRate = value; } }
        }
        public static double Probability
        {
            get { return probability / 100; }
            set 
            { 
                if (value > 1) { probability = value; }
                if(value > 100) { probability = 100; }
            }
        }
        public static double ARatio
        {
            get { return aRatio; }
            set { if (value > 0) { aRatio = value; } }
        }
        public static int InputSize { get { return layersFully.Layer[0].Input; } }
        public static int CountLayers { get { return layersConv.Layer.Count + layersFully.Layer.Count; } }
        public static int CountConv { get { return layersConv.Layer.Count; } }
        public static int CountFully { get { return layersFully.Layer.Count; } }
        public static int Answer { get { return layersFully.Layer[^1].Neurons.Max(); } }
        public static double Confidence { get { return Math.Round(layersFully.Layer[^1].Neurons.Percent(), commaConfidence) * 100; } }
        public static double Loss { get { return layersFully.Layer[^1].Loss; } }
        public static int Max // Count neurons last fully layers
        {
            get
            {
                if(selectionData.dataSet != null) { return selectionData.dataSet.Max(item => item.Right) + 1; }
                else { return 0; }
            }
        }
        public static string Error { get; internal set; }
        public static void AddIndex(AbLayer layer, int index = 0)
        {
            if (layer.GetType().BaseType == typeof(ConvalutionLayer)
                || layer.GetType() == typeof(ConvalutionLayer))
            {
                layersConv.Layer.Insert(index, (ConvalutionLayer)layer);
            }
            else { layersFully.Layer.Insert(index, (FullyConnectLayer)layer); }
        }
        public static void Add(AbLayer layer)
        {
            if (layer.GetType().BaseType == typeof(ConvalutionLayer)
                || layer.GetType() == typeof(ConvalutionLayer))
            {
                layersConv.Layer.Add((ConvalutionLayer)layer);
            }
            else { layersFully.Layer.Add((FullyConnectLayer)layer); }
        }
        public static void Load(AbLayer layer)
        {
            if (layer.GetType().BaseType == typeof(ConvalutionLayer)
                || layer.GetType() == typeof(ConvalutionLayer))
            {
                layersConv.Load((ConvalutionLayer)layer);
            }
            else { layersFully.Load((FullyConnectLayer)layer); }
            //CheckLoad();
        }
        public static void Remove(AbLayer layer)
        {
            if (layer.GetType().BaseType == typeof(ConvalutionLayer)
                || layer.GetType() == typeof(ConvalutionLayer))
            {
                int index = layersConv.Layer.IndexOf((ConvalutionLayer)layer);
                layersConv.Layer.RemoveAt(index);
            }
            else 
            {
                int index = layersFully.Layer.IndexOf((FullyConnectLayer)layer);
                layersFully.Layer.RemoveAt(index);
            }
        }
        public static void ForwardNet(Tensor input)
        {
            if (layersConv.Layer.Count != 0) { input = layersConv.DirectPassage(input); }
            layersFully.DirectPassage(input);
        }
        public static void BackNet(Tensor input)
        {
            input = layersFully.BackPassage(input, lRate, aRatio);
            if (layersConv.Layer.Count != 0) { layersConv.BackPassage(input, lRate, aRatio); }
        }
        public static void BackNoRefresh(Tensor input)
        {
            layersFully.Layer[^1].BackPropagation(input, input.Right, lRate, aRatio);
        }
        public static void DropOut() { layersFully.Drop(probability); }
        public static void DropIn() { layersFully.Unlock(); }
        public static void InitializationFully()
        {
            for (int i = 0; i < layersFully.Layer.Count; i++)
            {
                layersFully.Layer[i].Initialization();
            }
        }
        public static void InitializationConv()
        {
            if(selectionData != null && selectionData.Batches.Count >= 1)
            {
                Tensor input = selectionData.Batches[0][0];
                for (int i = 0; i < layersConv.Layer.Count; i++)
                {
                    layersConv.Layer[i].Initialization(input);
                    input = layersConv.Layer[i].Output;
                }
            }
        }
        public static void CheckLoad()
        {
            int saveInput = 0;
            if (layersFully.Layer.Count != 0) { saveInput = layersFully.Layer[0].Input; }
            if(selectionData != null)
            {
                if (selectionData.Batches.Count != 0 && Converter.IsLoad == true)
                {
                    InitializationConv();
                    if (layersConv.DirectPassage(selectionData.Batches[0][0]).FullSize != saveInput) 
                    { InitializationFullyOnlyInputAndClass(); }
                }
                else { InitializationConv(); InitializationFully(); }
            }
        }
        private static AbstractLayers<AbLayer> UnionLayers()
        {
            AbstractLayers<AbLayer> Layer = new AbstractLayers<AbLayer>();
            foreach(var item in layersConv.Layer)
            {
                Layer.Layer.Add(item);
            }
            foreach (var item in layersFully.Layer)
            {
                Layer.Layer.Add(item);
            }
            return Layer;
        }
        private static void SeparationLayers(AbstractLayers<AbLayer> Layer)
        {
            layersConv.Layer.Clear(); layersFully.Layer.Clear();
            foreach(var item in Layer.Layer)
            {
                if(item.GetType() == typeof(ConvalutionLayer) 
                    || item.GetType().BaseType == typeof(ConvalutionLayer))
                {
                    var fully = (ConvalutionLayer)item;
                    layersConv.Load(fully);
                }
                else 
                {
                    var conv = (FullyConnectLayer)item;
                    layersFully.Load(conv);
                }
            }
        }
        private static void InitializationFullyOnlyInputAndClass()
        {
            for (int i = 0; i < layersFully.Layer.Count; i++)
            {
                if (layersFully.Layer[i].GetType() == typeof(FullyConnectInput)
                    || layersFully.Layer[i].GetType() == typeof(FullyConnectClassifier))
                { layersFully.Layer[i].Initialization(); }
            }
        }
    }
}
