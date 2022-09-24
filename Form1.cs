using FIRC;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            var weights = Application.StartupPath + "\\FastestDet.onnx";
            var namesFile = Application.StartupPath + "\\coco.names";
            var jpgFile = Application.StartupPath + "\\dog.jpg";
            FastDet detector = new FastDet(weights, namesFile, 512);
            Mat img = new Mat(jpgFile);
            pictureBox1.Image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(detector.DrawImage(img, detector.InferenceImage(img)));
        }
    }
}
