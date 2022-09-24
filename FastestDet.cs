using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.Drawing;
using System.IO;

namespace FIRC
{

    public class DetectionResult
    {
        public int ClassId = -1;
        public string ClassName = null;
        public float Confidence = 0f;
        public int Xmin = 0;
        public int Ymin = 0;
        public int Xmax = 0;
        public int Ymax = 0;

    }

    public class FastDet
    {
        private Net net;
        public float threshold = 0.4f;
        public float nmsThreshold = 0.45f;
        public int left = 0;
        public int top = 0;
        public float scale = 0f;
        List<string> NameList = new List<string>();
        private OpenCvSharp.Size ModelSize;

        /// <summary>
        /// 从文件加载标签类别
        /// </summary>
        /// <param name="nameFile"></param>
        public void LoadLabels(string nameFile)
        {
            var data = File.ReadAllText(nameFile);
            data = data.TrimEnd((char[])"\n\r".ToCharArray());
            this.NameList = data.Split(new char[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries).ToList();

        }
        public FastDet(string weights, string namesFile, int modelSize =512)
        {
            ModelSize = new OpenCvSharp.Size(modelSize, modelSize);
            net = CvDnn.ReadNet(weights);
            net.SetPreferableBackend(Backend.OPENCV);
            net.SetPreferableTarget(0);
            LoadLabels(namesFile);

        }
        private int clip(int x1, int x2, int x3)
        {
            var x4 = Math.Min(x1, x3);
            return Math.Max(x2, x4);

        }
        private List<DetectionResult> ScaleCoordinates(List<DetectionResult> drList, float pad_w, float pad_h,
                                        float scale, int factWidth, int factHeight)
        {
            if (drList.Count == 0)
            {
                return drList;
            }
            for (int i = 0; i < drList.Count; i++)
            {
                drList[i].Xmin = Convert.ToInt32((drList[i].Xmin - pad_w) / scale);  // x padding
                drList[i].Ymin = Convert.ToInt32((drList[i].Ymin - pad_h) / scale);  // y padding
                drList[i].Xmax = Convert.ToInt32((drList[i].Xmax - pad_w) / scale);  // x padding
                drList[i].Ymax = Convert.ToInt32((drList[i].Ymax - pad_h) / scale);  // y padding

                drList[i].Xmin = clip(drList[i].Xmin, 0, factWidth);
                drList[i].Ymin = clip(drList[i].Ymin, 0, factHeight);
                drList[i].Xmax = clip(drList[i].Xmax, 0, factWidth);
                drList[i].Ymax = clip(drList[i].Ymax, 0, factHeight);

            }
            return drList;
        }
        /// <summary>
        /// 对齐方式代码改自https://github.com/yasenh/libtorch-yolov5
        /// </summary>
        /// <param name="src"></param>
        /// <param name="modelSize"></param>
        /// <returns></returns>
        private Mat LetterBoxImage(Mat src, OpenCvSharp.Size modelSize)
        {

            var in_h = src.Height;
            var in_w = src.Width;
            float out_h = modelSize.Height;
            float out_w = modelSize.Width;

            scale = Math.Min(out_w / in_w, out_h / in_h);

            int mid_h = Convert.ToInt32(in_h * scale);
            int mid_w = Convert.ToInt32(in_w * scale);
            Mat dest = new Mat();
            Cv2.Resize(src, dest, new OpenCvSharp.Size(mid_w, mid_h));

            top = Convert.ToInt32((out_h - mid_h) / 2);
            int down = Convert.ToInt32((out_h - mid_h + 1) / 2);
            left = Convert.ToInt32((out_w - mid_w) / 2);
            int right = Convert.ToInt32((out_w - mid_w + 1) / 2);
            Cv2.CopyMakeBorder(dest, dest, top, down, left, right, BorderTypes.Constant, new Scalar(114, 114, 114));
            return dest;

        }

        public Mat DrawImage(Mat bmp, List<DetectionResult> drList)
        {
            if (drList.Count == 0)
            {
                return bmp;
            }
            for (int i = 0; i < drList.Count; i++)
            {
                OpenCvSharp.Rect rect = new OpenCvSharp.Rect(drList[i].Xmin, drList[i].Ymin, drList[i].Xmax - drList[i].Xmin, drList[i].Ymax - drList[i].Ymin);
                Cv2.Rectangle(bmp, rect, new Scalar(0, 255, 0), 3);
                Cv2.PutText(bmp, drList[i].ClassName + " " + drList[i].Confidence.ToString("F2"), new OpenCvSharp.Point(drList[i].Xmin - 10, drList[i].Ymin), HersheyFonts.HersheySimplex, 1.0, new Scalar(255, 0, 0), 3);
            }

            return bmp;

        }
        double sigmoid(double x)
        {
            return 1.0 / (1 + Math.Exp(-x));
        }

        public List<DetectionResult> InferenceImage(Mat frame)
        {
            var fact_w = frame.Width;
            var fact_h = frame.Height;
            List<DetectionResult> drList = new List<DetectionResult>();
            try
            {
                frame = LetterBoxImage(frame, ModelSize);
                //生成blob, 块尺寸可以是320/416/608
                var blob = CvDnn.BlobFromImage(frame, 1.0 / 255, ModelSize, new Scalar(), true, false);
                // 输入数据
                net.SetInput(blob);

                //获得输出层名
                var outNames = net.GetUnconnectedOutLayersNames();

                //转换成 Mat[]
                var outs = outNames.Select(_ => new Mat()).ToArray();

                net.Forward(outs, outNames);
                Console.WriteLine(outs.Length);
                int num_proposal = outs[0].Height;
                int nout = outs[0].Cols;
                List<float> confidences=new List<float>();
                List<Rect> boxes=new List<Rect>();
                List<int> classIds=new List<int>();
                int  row_ind = 0; ///box_score, xmin,ymin,xamx,ymax,class_score
                const int num_grid_x = 32;
                const int num_grid_y = 32;
                for (int i = 0; i < num_grid_y; i++)
                {
                    for (int j = 0; j < num_grid_x; j++)
                    {
                        Mat scores = outs[0].Row(row_ind).ColRange(5, nout);
                        // Get the value and location of the maximum score
                        Cv2.MinMaxLoc(scores, out double min_class_socre, out double max_class_socre, out OpenCvSharp.Point min, out OpenCvSharp.Point max);
                        max_class_socre *= outs[0].At<float>(row_ind, 0);
                        if (max_class_socre > this.threshold)
                        {
                            int class_idx = max.X;
                            var cx = (Math.Tanh(outs[0].At<float>(row_ind, 1)) + j) / num_grid_x;  
                            var cy = (Math.Tanh(outs[0].At<float>(row_ind, 2)) + i) / (float)num_grid_y;   ///cy
                            var w = sigmoid(outs[0].At<float>(row_ind, 3));   ///w
                            var h = sigmoid(outs[0].At<float>(row_ind, 4));  ///h

                            cx *= frame.Cols;
                            cy *= frame.Rows;
                            w *= frame.Cols;
                            h *= frame.Rows;

                            int left = Convert.ToInt32(cx - 0.5 * w);
                            int top = Convert.ToInt32(cy - 0.5 * h);

                            confidences.Add((float)max_class_socre);
                            boxes.Add(new Rect(left, top, Convert.ToInt32(w), Convert.ToInt32(h)));
                            classIds.Add(class_idx);
                        }
                        row_ind++;
                    }
                }


                int[] indices;
                CvDnn.NMSBoxes(boxes, confidences, this.threshold, this.nmsThreshold, out indices);
                for (int i = 0; i < indices.Length; ++i)
                {
                    int idx = indices[i];
                    Rect box = boxes[idx];
                    if (confidences[idx] > this.threshold)
                    {
                        DetectionResult dr = new DetectionResult();
                        dr.ClassId = classIds[idx];
                        dr.Confidence = confidences[idx];
                        dr.ClassName = this.NameList[dr.ClassId];
                        dr.Xmin = box.X;
                        dr.Ymin = box.Y;
                        dr.Xmax = box.X + box.Width;
                        dr.Ymax = box.Y + box.Height;
                        drList.Add(dr);
                    }
                }

                drList = ScaleCoordinates(drList, left, top, scale, fact_w, fact_h);
                return drList;

            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message.ToString());
                return drList;
            }
        }





    }
}
