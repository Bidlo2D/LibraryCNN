using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LibraryCNN
{
    [Serializable]
    public class Neuron : IComparable<Neuron>
    {
        public bool block;
        private double active;
        public double Value 
        { 
            get
            {
                return block == true ? 0 : active;
            }
            set { active = value; }
        }
        public Neuron() { }
        public Neuron(double value) { Value = value; }
        public int CompareTo(Neuron o) { return Value.CompareTo(o.Value); }
    }
}
