using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ScottPlot;

namespace XOR_NeuralNetwork_LR3
{
    public enum ActivationType { Sigmoid, ReLU }

    public class NeuralNetwork
    {
        private readonly int inputSize;
        private readonly int hiddenSize;

        private readonly double[,] weightsInputHidden;
        private readonly double[] biasHidden;
        private readonly double[] hiddenSums;
        private readonly double[] hiddenOutputs;

        private readonly double[] weightsHiddenOutput;
        private double biasOutput;

        private readonly Random rnd;

        public ActivationType HiddenActivation { get; set; }

        private const double LeakyAlpha = 0.01;

        public NeuralNetwork(int input, int hidden, int output,
                             int seed = 42,
                             ActivationType hiddenAct = ActivationType.Sigmoid)
        {
            inputSize = input;
            hiddenSize = hidden;
            HiddenActivation = hiddenAct;
            rnd = new Random(seed);

            weightsInputHidden = new double[inputSize, hiddenSize];
            biasHidden = new double[hiddenSize];
            hiddenSums = new double[hiddenSize];
            hiddenOutputs = new double[hiddenSize];

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < hiddenSize; j++)
                    weightsInputHidden[i, j] = (rnd.NextDouble() - 0.5) * 2.0;

            for (int j = 0; j < hiddenSize; j++)
            {
                if (HiddenActivation == ActivationType.ReLU)
                    biasHidden[j] = 0.5 + rnd.NextDouble();
                else
                    biasHidden[j] = (rnd.NextDouble() - 0.5) * 2.0;
            }

            weightsHiddenOutput = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
                weightsHiddenOutput[j] = (rnd.NextDouble() - 0.5) * 2.0;

            biasOutput = (rnd.NextDouble() - 0.5) * 2.0;
        }

        private static double Sigmoid(double x)
        {
            if (x < -45) return 0.0;
            if (x > 45) return 1.0;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private static double SigmoidDeriv(double y) => y * (1.0 - y);

        private static double ReLU(double x) => x > 0.0 ? x : LeakyAlpha * x;

        private double Activate(double sum)
            => HiddenActivation == ActivationType.Sigmoid ? Sigmoid(sum) : ReLU(sum);

        private double ActivateDeriv(double sum, double output)
        {
            if (HiddenActivation == ActivationType.Sigmoid)
                return SigmoidDeriv(output);
            return sum > 0.0 ? 1.0 : LeakyAlpha;
        }

        public double Forward(double[] inputs)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                double sum = biasHidden[j];
                for (int i = 0; i < inputSize; i++)
                    sum += inputs[i] * weightsInputHidden[i, j];

                hiddenSums[j] = sum;
                hiddenOutputs[j] = Activate(sum);
            }

            double finalSum = biasOutput;
            for (int j = 0; j < hiddenSize; j++)
                finalSum += hiddenOutputs[j] * weightsHiddenOutput[j];

            return Sigmoid(finalSum);
        }

        public double TrainBCE(double[] inputs, double targetNorm, double learningRate)
        {
            double output = Forward(inputs);

            double outputDelta = output - targetNorm;

            double[] hiddenDeltas = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
            {
                double errHidden = outputDelta * weightsHiddenOutput[j];
                hiddenDeltas[j] = errHidden * ActivateDeriv(hiddenSums[j], hiddenOutputs[j]);
            }

            for (int j = 0; j < hiddenSize; j++)
                weightsHiddenOutput[j] -= learningRate * outputDelta * hiddenOutputs[j];
            biasOutput -= learningRate * outputDelta;

            for (int j = 0; j < hiddenSize; j++)
            {
                for (int i = 0; i < inputSize; i++)
                    weightsInputHidden[i, j] -= learningRate * hiddenDeltas[j] * inputs[i];
                biasHidden[j] -= learningRate * hiddenDeltas[j];
            }

            const double eps = 1e-12;
            double yc = Math.Min(Math.Max(output, eps), 1.0 - eps);
            return -(targetNorm * Math.Log(yc) + (1.0 - targetNorm) * Math.Log(1.0 - yc));
        }
    }

    public class TrainResult
    {
        public int Seed;
        public int Epochs;
        public double FinalError;
        public double Accuracy;
        public double MAE;
        public bool Converged;
        public List<double> History = new List<double>();
        public NeuralNetwork Net;
    }

    class Program
    {
        const double C0 = 0.0;
        const double C1 = -6.0;

        static readonly double TargetMin = Math.Min(C0, C1);
        static readonly double TargetMax = Math.Max(C0, C1);
        static readonly double TargetRange = TargetMax - TargetMin;

        static double NormalizeInput(double x) => x / 10.0;
        static double NormalizeTarget(double t) => (t - TargetMin) / TargetRange;
        static double DenormalizeTarget(double y) => TargetMin + y * TargetRange;

        static readonly double[][] rawInputs =
        {
            new double[] {  0,  0 },
            new double[] {  0, -6 },
            new double[] { -6,  0 },
            new double[] { -6, -6 }
        };
        static readonly double[] rawTargets = { C0, C1, C1, C0 };

        static double[][] inputs;
        static double[] targetsNorm;

        const double LrSigmoid = 2.0;
        const double LrReLU = 0.1;

        const double Ee = 0.01;
        const int SafetyLimit = 200000;

        static void Main()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            inputs = new double[rawInputs.Length][];
            targetsNorm = new double[rawTargets.Length];

            Console.WriteLine("Лабораторная работа 3: MLP 2-2-1, XOR, Sigmoid и ReLU");
            Console.WriteLine($"Вариант 1: c0 = {C0}, c1 = {C1}");
            Console.WriteLine();

            for (int i = 0; i < rawInputs.Length; i++)
            {
                inputs[i] = new double[] { NormalizeInput(rawInputs[i][0]),
                                           NormalizeInput(rawInputs[i][1]) };
                targetsNorm[i] = NormalizeTarget(rawTargets[i]);
            }

            int[] seeds = { 1, 7, 21, 42, 123 };
            var resultsA = new List<TrainResult>();
            var resultsB = new List<TrainResult>();

            foreach (int seed in seeds)
            {
                var mlp = new NeuralNetwork(2, 2, 1, seed, ActivationType.Sigmoid);
                var res = TrainUntilConvergence(mlp, LrSigmoid);
                res.Seed = seed;
                res.Net = mlp;
                Evaluate(res);
                resultsA.Add(res);
            }

            foreach (int seed in seeds)
            {
                var mlp = new NeuralNetwork(2, 2, 1, seed, ActivationType.ReLU);
                var res = TrainUntilConvergence(mlp, LrReLU);
                res.Seed = seed;
                res.Net = mlp;
                Evaluate(res);
                resultsB.Add(res);
            }

            PrintSummaryTable("Конфигурация A (Sigmoid, BCE)", resultsA);
            PrintSummaryTable("Конфигурация B (ReLU, BCE)", resultsB);
            PrintStability(resultsA, resultsB);

            var repA = resultsA.First(r => r.Converged);
            var repB = resultsB.First(r => r.Converged);

            Console.WriteLine("\nРезультаты представительных экземпляров");
            Console.WriteLine("\nКонфигурация A (Sigmoid):");
            PrintResults(repA.Net);
            Console.WriteLine("\nКонфигурация B (ReLU):");
            PrintResults(repB.Net);

            GeneratePlots(resultsA, resultsB, repA, repB);
            RunInteractiveMode(repA.Net, repB.Net);
        }

        static TrainResult TrainUntilConvergence(NeuralNetwork net, double lr)
        {
            var res = new TrainResult();
            int epoch = 0;
            double totalError = double.MaxValue;

            while (totalError > Ee && epoch < SafetyLimit)
            {
                totalError = 0.0;
                for (int i = 0; i < inputs.Length; i++)
                    totalError += net.TrainBCE(inputs[i], targetsNorm[i], lr);

                res.History.Add(totalError);
                epoch++;
            }

            res.Epochs = epoch;
            res.FinalError = totalError;
            res.Converged = totalError <= Ee;
            return res;
        }

        static void Evaluate(TrainResult res)
        {
            int correct = 0;
            double sumAbs = 0.0;

            for (int i = 0; i < inputs.Length; i++)
            {
                double yNorm = res.Net.Forward(inputs[i]);
                double yRaw = DenormalizeTarget(yNorm);

                double predClass = Math.Abs(yRaw - C0) < Math.Abs(yRaw - C1) ? C0 : C1;
                if (Math.Abs(predClass - rawTargets[i]) < 1e-9) correct++;
                sumAbs += Math.Abs(yRaw - rawTargets[i]);
            }

            res.Accuracy = (double)correct / inputs.Length;
            res.MAE = sumAbs / inputs.Length;
        }

        static void PrintSummaryTable(string title, List<TrainResult> results)
        {
            Console.WriteLine($"\n{title}");
            Console.WriteLine($"{"Seed",6} | {"Эпохи",8} | {"Итог.Es",12} | " +
                              $"{"Acc",6} | {"MAE",8}");
            foreach (var r in results)
                Console.WriteLine($"{r.Seed,6} | {r.Epochs,8} | {r.FinalError,12:F6} | " +
                                  $"{r.Accuracy,6:F2} | {r.MAE,8:F4}");
        }

        static void PrintStability(List<TrainResult> A, List<TrainResult> B)
        {
            Console.WriteLine("\nУстойчивость сходимости, 5 запусков");
            foreach (var pair in new[] { Tuple.Create("A (Sigmoid)", A),
                                         Tuple.Create("B (ReLU)",    B) })
            {
                string name = pair.Item1;
                var list = pair.Item2;
                int conv = list.Count(r => r.Converged);
                var eps = list.Where(r => r.Converged).Select(r => r.Epochs).ToList();

                if (eps.Count > 0)
                {
                    double mean = eps.Average();
                    double std = Math.Sqrt(eps.Select(e => (e - mean) * (e - mean)).Average());
                    Console.WriteLine($"{name}: сошлось {conv}/{list.Count}, " +
                                      $"эпохи [{eps.Min()}..{eps.Max()}], mean={mean:F1}, std={std:F1}");
                }
                else
                    Console.WriteLine($"{name}: сошлось 0/{list.Count}");
            }
        }

        static void PrintResults(NeuralNetwork net)
        {
            Console.WriteLine($"{"A",6} {"B",6} {"Ожид",7} {"out_norm",9} {"out_raw",10} {"Класс",7} {"Вердикт",9}");
            for (int i = 0; i < inputs.Length; i++)
            {
                double yNorm = net.Forward(inputs[i]);
                double yRaw = DenormalizeTarget(yNorm);
                double cls = NearestClass(yRaw);
                bool ok = Math.Abs(cls - rawTargets[i]) < 1e-9;

                Console.WriteLine($"{rawInputs[i][0],6} {rawInputs[i][1],6} {rawTargets[i],7} " +
                                  $"{yNorm,9:F4} {yRaw,10:F4} {cls,7:F0} " +
                                  $"{(ok ? "Верно" : "Неверно"),9}");
            }
        }

        static double HeuristicExpected(double a, double b)
        {
            int classA = Math.Abs(a - C0) < Math.Abs(a - C1) ? 0 : 1;
            int classB = Math.Abs(b - C0) < Math.Abs(b - C1) ? 0 : 1;
            return ((classA ^ classB) == 0) ? C0 : C1;
        }

        static double NearestClass(double yRaw)
            => Math.Abs(yRaw - C0) < Math.Abs(yRaw - C1) ? C0 : C1;

        static void RunInteractiveMode(NeuralNetwork netA, NeuralNetwork netB)
        {
            Console.WriteLine("\nРежим функционирования сети");
            Console.WriteLine("Ввод A B из [-10; 10], q - выход.\n");

            Console.WriteLine("4 примера из обучающей выборки:");
            for (int i = 0; i < rawInputs.Length; i++)
            {
                Console.WriteLine($"Пример {i + 1}: A={rawInputs[i][0]}, B={rawInputs[i][1]}, " +
                                  $"ожидание={rawTargets[i]}");
                PrintPrediction(netA, rawInputs[i][0], rawInputs[i][1], rawTargets[i], "A");
                PrintPrediction(netB, rawInputs[i][0], rawInputs[i][1], rawTargets[i], "B");
            }

            Console.WriteLine("\nДополнительные точки:");
            double[][] extra =
            {
                new double[] {  5,  5 },
                new double[] { -3,  7 },
                new double[] {  2, -8 },
                new double[] { -9, -2 }
            };
            foreach (var p in extra)
            {
                double expected = HeuristicExpected(p[0], p[1]);
                Console.WriteLine($"Доп. точка: A={p[0]}, B={p[1]}, ожидание={expected}");
                PrintPrediction(netA, p[0], p[1], expected, "A");
                PrintPrediction(netB, p[0], p[1], expected, "B");
            }

            Console.WriteLine("\nИнтерактивный режим:");
            while (true)
            {
                Console.Write("A B > ");
                string line = Console.ReadLine();
                if (line == null || line.Trim().ToLower() == "q") break;

                var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length != 2) continue;

                if (double.TryParse(parts[0], out double a) &&
                    double.TryParse(parts[1], out double b))
                {
                    if (a < -10 || a > 10 || b < -10 || b > 10)
                    {
                        Console.WriteLine("Значения должны быть в [-10; 10].");
                        continue;
                    }
                    double expected = HeuristicExpected(a, b);
                    PrintPrediction(netA, a, b, expected, "A");
                    PrintPrediction(netB, a, b, expected, "B");
                }
            }
        }

        static void PrintPrediction(NeuralNetwork net, double a, double b,
                                    double? expected, string tag)
        {
            double[] testInput = { NormalizeInput(a), NormalizeInput(b) };
            double yNorm = net.Forward(testInput);
            double yRaw = DenormalizeTarget(yNorm);
            double cls = NearestClass(yRaw);

            string expStr = expected.HasValue ? expected.Value.ToString("F0") : "-";
            string verdict = "";
            if (expected.HasValue)
                verdict = Math.Abs(cls - expected.Value) < 1e-9 ? "Верно" : "Неверно";

            Console.WriteLine($"  [{tag}] A={a,5}, B={b,5}, out_norm={yNorm:F4}, " +
                              $"out_raw={yRaw,8:F4}, класс={cls,5:F0}, " +
                              $"ожид={expStr,5} {verdict}");
        }

        static string GetProjectDir()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            return Path.GetFullPath(Path.Combine(baseDir, @"..\..\..\"));
        }

        static void GeneratePlots(List<TrainResult> resultsA, List<TrainResult> resultsB,
                                  TrainResult repA, TrainResult repB)
        {
            string dir = GetProjectDir();

            var plotConv = new Plot();
            plotConv.Title("Сходимость: Sigmoid и ReLU на скрытом слое, XOR");
            plotConv.XLabel("Номер эпохи");
            plotConv.YLabel("Суммарная ошибка Es, BCE");

            double[] epA = Enumerable.Range(0, repA.History.Count).Select(i => (double)i).ToArray();
            double[] epB = Enumerable.Range(0, repB.History.Count).Select(i => (double)i).ToArray();

            var scA = plotConv.Add.Scatter(epA, repA.History.ToArray());
            scA.LegendText = $"A (Sigmoid): {repA.Epochs} эпох";
            scA.Color = Colors.Blue;
            scA.LineWidth = 2;

            var scB = plotConv.Add.Scatter(epB, repB.History.ToArray());
            scB.LegendText = $"B (ReLU): {repB.Epochs} эпох";
            scB.Color = Colors.Red;
            scB.LineWidth = 2;

            var lineEe = plotConv.Add.HorizontalLine(Ee);
            lineEe.Color = Colors.Green;
            lineEe.LinePattern = LinePattern.Dashed;
            lineEe.LegendText = $"Ee = {Ee}";

            plotConv.ShowLegend();
            plotConv.SavePng(Path.Combine(dir, "plot_convergence.png"), 900, 600);

            var plotEpochs = new Plot();
            plotEpochs.Title("Число эпох до сходимости по 5 запускам");
            plotEpochs.XLabel("Seed");
            plotEpochs.YLabel("Эпохи");

            double[] seeds = resultsA.Select(r => (double)r.Seed).ToArray();
            double[] epAD = resultsA.Select(r => r.Converged ? (double)r.Epochs : SafetyLimit).ToArray();
            double[] epBD = resultsB.Select(r => r.Converged ? (double)r.Epochs : SafetyLimit).ToArray();

            var barsA = plotEpochs.Add.Bars(seeds.Select(s => s - 1.5).ToArray(), epAD);
            barsA.LegendText = "A (Sigmoid)";
            barsA.Color = Colors.Blue;

            var barsB = plotEpochs.Add.Bars(seeds.Select(s => s + 1.5).ToArray(), epBD);
            barsB.LegendText = "B (ReLU)";
            barsB.Color = Colors.Red;

            plotEpochs.ShowLegend();
            plotEpochs.SavePng(Path.Combine(dir, "plot_epochs.png"), 900, 600);

            var multiplot = new Multiplot();
            multiplot.AddPlots(2);

            GenerateSurfacePlot(multiplot.GetPlot(0), repA.Net, "Конфигурация A (Sigmoid)");
            GenerateSurfacePlot(multiplot.GetPlot(1), repB.Net, "Конфигурация B (ReLU)");

            multiplot.SavePng(Path.Combine(dir, "plot_surfaces.png"), 1400, 600);

            var plotMae = new Plot();
            plotMae.Title("MAE в исходной шкале c0..c1");
            plotMae.XLabel("Seed");
            plotMae.YLabel("MAE");

            double[] maeAD = resultsA.Select(r => r.MAE).ToArray();
            double[] maeBD = resultsB.Select(r => r.MAE).ToArray();

            var barsMaeA = plotMae.Add.Bars(seeds.Select(s => s - 1.5).ToArray(), maeAD);
            barsMaeA.LegendText = "A (Sigmoid)";
            barsMaeA.Color = Colors.Blue;

            var barsMaeB = plotMae.Add.Bars(seeds.Select(s => s + 1.5).ToArray(), maeBD);
            barsMaeB.LegendText = "B (ReLU)";
            barsMaeB.Color = Colors.Red;

            plotMae.ShowLegend();
            plotMae.SavePng(Path.Combine(dir, "plot_mae.png"), 900, 600);

            Console.WriteLine("\nСохранены графики:");
            Console.WriteLine("  plot_convergence.png");
            Console.WriteLine("  plot_epochs.png");
            Console.WriteLine("  plot_surfaces.png");
            Console.WriteLine("  plot_mae.png");
        }

        static void GenerateSurfacePlot(Plot plot, NeuralNetwork net, string title)
        {
            const int gridSize = 80;
            double[,] zData = new double[gridSize, gridSize];

            for (int i = 0; i < gridSize; i++)
            {
                for (int j = 0; j < gridSize; j++)
                {
                    double a = -10 + 20.0 * i / (gridSize - 1);
                    double b = -10 + 20.0 * j / (gridSize - 1);
                    double[] inp = { NormalizeInput(a), NormalizeInput(b) };
                    zData[j, i] = DenormalizeTarget(net.Forward(inp));
                }
            }

            var heatmap = plot.Add.Heatmap(zData);
            heatmap.Extent = new CoordinateRect(-10, 10, -10, 10);
            heatmap.FlipVertically = true;
            heatmap.Colormap = new ScottPlot.Colormaps.Turbo();

            for (int k = 0; k < rawInputs.Length; k++)
            {
                var marker = plot.Add.Marker(rawInputs[k][0], rawInputs[k][1]);
                marker.Color = rawTargets[k] == C0 ? Colors.Blue : Colors.Red;
                marker.Size = 15;
                marker.Shape = MarkerShape.FilledCircle;
            }

            plot.Title(title);
            plot.XLabel("A");
            plot.YLabel("B");
            plot.Axes.SetLimits(-10, 10, -10, 10);
        }
    }
}