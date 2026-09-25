using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ScottPlot;

namespace XOR_NeuralNetwork
{
    public class NeuralNetwork
    {
        private int inputSize;
        private int hiddenSize;
        private int outputSize;

        private double[,] weightsInputHidden;
        private double[] biasHidden;
        private double[] hiddenOutputs;

        private double[] weightsHiddenOutput;
        private double biasOutput;

        private Random rnd;

        public NeuralNetwork(int input, int hidden, int output, int seed = 42)
        {
            inputSize = input;
            hiddenSize = hidden;
            outputSize = output;
            rnd = new Random(seed);

            if (hiddenSize > 0)
            {
                weightsInputHidden = new double[inputSize, hiddenSize];
                biasHidden = new double[hiddenSize];
                hiddenOutputs = new double[hiddenSize];
                InitializeArray(weightsInputHidden);
                InitializeArray(biasHidden);
            }

            weightsHiddenOutput = new double[hiddenSize > 0 ? hiddenSize : inputSize];
            biasOutput = 0;
            InitializeArray(weightsHiddenOutput);
        }

        private void InitializeArray(double[,] array)
        {
            for (int i = 0; i < array.GetLength(0); i++)
                for (int j = 0; j < array.GetLength(1); j++)
                    array[i, j] = rnd.NextDouble() - 0.5;
        }

        private void InitializeArray(double[] array)
        {
            for (int i = 0; i < array.Length; i++)
                array[i] = rnd.NextDouble() - 0.5;
        }

        private double Sigmoid(double x)
        {
            if (x < -45) return 0;
            if (x > 45) return 1;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double output)
        {
            return output * (1.0 - output);
        }

        public double Forward(double[] inputs)
        {
            if (hiddenSize > 0)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    double sum = biasHidden[j];
                    for (int i = 0; i < inputSize; i++)
                        sum += inputs[i] * weightsInputHidden[i, j];
                    hiddenOutputs[j] = Sigmoid(sum);
                }

                double finalSum = biasOutput;
                for (int j = 0; j < hiddenSize; j++)
                    finalSum += hiddenOutputs[j] * weightsHiddenOutput[j];
                return Sigmoid(finalSum);
            }
            else
            {
                double sum = biasOutput;
                for (int i = 0; i < inputSize; i++)
                    sum += inputs[i] * weightsHiddenOutput[i];
                return Sigmoid(sum);
            }
        }

        public double TrainMSE(double[] inputs, double targetNorm, double learningRate)
        {
            double output = Forward(inputs);
            double error = output - targetNorm;
            double outputDelta = error * SigmoidDerivative(output);

            if (hiddenSize > 0)
            {
                for (int j = 0; j < hiddenSize; j++)
                    weightsHiddenOutput[j] -= learningRate * outputDelta * hiddenOutputs[j];
                biasOutput -= learningRate * outputDelta;

                double[] hiddenDeltas = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++)
                {
                    double errorHidden = outputDelta * weightsHiddenOutput[j];
                    hiddenDeltas[j] = errorHidden * SigmoidDerivative(hiddenOutputs[j]);
                }

                for (int j = 0; j < hiddenSize; j++)
                {
                    for (int i = 0; i < inputSize; i++)
                        weightsInputHidden[i, j] -= learningRate * hiddenDeltas[j] * inputs[i];
                    biasHidden[j] -= learningRate * hiddenDeltas[j];
                }
            }
            else
            {
                for (int i = 0; i < inputSize; i++)
                    weightsHiddenOutput[i] -= learningRate * outputDelta * inputs[i];
                biasOutput -= learningRate * outputDelta;
            }

            return error * error;
        }

        public double TrainBCE(double[] inputs, double targetNorm, double learningRate)
        {
            double output = Forward(inputs);
            double outputDelta = output - targetNorm;

            if (hiddenSize > 0)
            {
                for (int j = 0; j < hiddenSize; j++)
                    weightsHiddenOutput[j] -= learningRate * outputDelta * hiddenOutputs[j];
                biasOutput -= learningRate * outputDelta;

                double[] hiddenDeltas = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++)
                {
                    double errorHidden = outputDelta * weightsHiddenOutput[j];
                    hiddenDeltas[j] = errorHidden * SigmoidDerivative(hiddenOutputs[j]);
                }

                for (int j = 0; j < hiddenSize; j++)
                {
                    for (int i = 0; i < inputSize; i++)
                        weightsInputHidden[i, j] -= learningRate * hiddenDeltas[j] * inputs[i];
                    biasHidden[j] -= learningRate * hiddenDeltas[j];
                }
            }
            else
            {
                for (int i = 0; i < inputSize; i++)
                    weightsHiddenOutput[i] -= learningRate * outputDelta * inputs[i];
                biasOutput -= learningRate * outputDelta;
            }

            double eps = 1e-12;
            double yc = Math.Min(Math.Max(output, eps), 1.0 - eps);
            return -(targetNorm * Math.Log(yc) + (1.0 - targetNorm) * Math.Log(1.0 - yc));
        }

        public double Train(double[] inputs, double targetNorm, double learningRate)
            => TrainMSE(inputs, targetNorm, learningRate);
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

        static double Normalize(double x) => (x + 10.0) / 20.0;
        static double NormalizeTarget(double t) => (t - TargetMin) / TargetRange;
        static double DenormalizeTarget(double y) => TargetMin + y * TargetRange;

        static readonly double[][] rawInputs = new double[][]
        {
            new double[] {  0,  0 },
            new double[] {  0, -6 },
            new double[] { -6,  0 },
            new double[] { -6, -6 }
        };
        static readonly double[] rawTargets = new double[] { C0, C1, C1, C0 };

        static double[][] inputs;
        static double[] targetsNorm;

        const double learningRate = 2.0;
        const double EeMSE = 0.01;
        const double EeBCE = 0.01;
        const int safetyLimit = 200000;

        static void Main(string[] args)
        {
            inputs = new double[rawInputs.Length][];
            targetsNorm = new double[rawTargets.Length];

            Console.WriteLine("Min-max нормализация целевых значений");
            Console.WriteLine($"Шкала [c0;c1] = [{C0}; {C1}]  min={TargetMin}, max={TargetMax}, range={TargetRange}");
            Console.WriteLine($"c0 = {C0} -> t_norm = {NormalizeTarget(C0):F4}");
            Console.WriteLine($"c1 = {C1} -> t_norm = {NormalizeTarget(C1):F4}");
            Console.WriteLine();

            for (int i = 0; i < rawInputs.Length; i++)
            {
                inputs[i] = new double[] { Normalize(rawInputs[i][0]), Normalize(rawInputs[i][1]) };
                targetsNorm[i] = NormalizeTarget(rawTargets[i]);
            }

            Console.WriteLine("Нормализованные данные:");
            Console.WriteLine($"{"A_raw",7} {"B_raw",7} | {"A_norm",7} {"B_norm",7} | {"t_raw",6} {"t_norm",7}");
            for (int i = 0; i < inputs.Length; i++)
                Console.WriteLine($"{rawInputs[i][0],7:F1} {rawInputs[i][1],7:F1} | " +
                                  $"{inputs[i][0],7:F4} {inputs[i][1],7:F4} | " +
                                  $"{rawTargets[i],6:F1} {targetsNorm[i],7:F4}");

            int[] seeds = { 1, 7, 21, 42, 123 };
            var resultsA = new List<TrainResult>();
            var resultsB = new List<TrainResult>();

            Console.WriteLine("\nКОНФИГУРАЦИЯ А (MSE на нормализованных 0/1)");
            foreach (int seed in seeds)
            {
                var mlp = new NeuralNetwork(2, 2, 1, seed);
                var res = TrainUntilConvergence(mlp, inputs, targetsNorm,
                                                learningRate, EeMSE, "MLP-MSE",
                                                safetyLimit, mode: "mse");
                res.Seed = seed;
                res.Net = mlp;
                Evaluate(res, mlp, mode: "mse");
                resultsA.Add(res);
                PrintResult("A", res);
            }

            Console.WriteLine("\nКОНФИГУРАЦИЯ Б (BCE на нормализованных 0/1)");
            foreach (int seed in seeds)
            {
                var mlp = new NeuralNetwork(2, 2, 1, seed);
                var res = TrainUntilConvergence(mlp, inputs, targetsNorm,
                                                learningRate, EeBCE, "MLP-BCE",
                                                safetyLimit, mode: "bce");
                res.Seed = seed;
                res.Net = mlp;
                Evaluate(res, mlp, mode: "bce");
                resultsB.Add(res);
                PrintResult("B", res);
            }

            PrintSummaryTable("Конфигурация А (MSE, нормализованные 0/1)", resultsA);
            PrintSummaryTable("Конфигурация Б (BCE, нормализованные 0/1)", resultsB);
            PrintStability(resultsA, resultsB);

            var repA = resultsA.First(r => r.Converged);
            var repB = resultsB.First(r => r.Converged);

            Console.WriteLine("\nИтоговые результаты");
            Console.WriteLine("\nMLP (2-2-1), Конфигурация А (MSE):");
            PrintResults(repA.Net, inputs, rawInputs, rawTargets, mode: "mse");

            Console.WriteLine("\nMLP (2-2-1), Конфигурация Б (BCE):");
            PrintResults(repB.Net, inputs, rawInputs, rawTargets, mode: "bce");

            GeneratePlots(resultsA, resultsB, repA, repB);

            RunInteractiveMode(repB.Net);
        }

        static TrainResult TrainUntilConvergence(NeuralNetwork net, double[][] inputs,
            double[] targetsNorm, double lr, double targetError, string name,
            int limit, string mode)
        {
            var res = new TrainResult();
            int epoch = 0;
            double totalError = double.MaxValue;

            while (totalError > targetError && epoch < limit)
            {
                totalError = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    if (mode == "mse")
                        totalError += net.TrainMSE(inputs[i], targetsNorm[i], lr);
                    else
                        totalError += net.TrainBCE(inputs[i], targetsNorm[i], lr);
                }

                res.History.Add(totalError);

                if (epoch % 1000 == 0)
                    Console.WriteLine($"{name} - Эпоха: {epoch}, Ошибка: {totalError:F6}");

                epoch++;
            }

            if (totalError <= targetError)
                Console.WriteLine($"{name} Сходимость на эпохе {epoch}. Ошибка: {totalError:F6}");
            else
                Console.WriteLine($"{name} Достигнут лимит {limit} эпох. Ошибка: {totalError:F6}");

            res.Epochs = epoch;
            res.FinalError = totalError;
            res.Converged = totalError <= targetError;
            return res;
        }

        static void Evaluate(TrainResult res, NeuralNetwork net, string mode)
        {
            int correct = 0;
            double sumAbs = 0.0;

            for (int i = 0; i < inputs.Length; i++)
            {
                double yNorm = net.Forward(inputs[i]);
                double yRaw = DenormalizeTarget(yNorm);

                double predClass = Math.Abs(yRaw - C0) < Math.Abs(yRaw - C1) ? C0 : C1;
                if (Math.Abs(predClass - rawTargets[i]) < 1e-9) correct++;
                sumAbs += Math.Abs(yRaw - rawTargets[i]);
            }

            res.Accuracy = (double)correct / inputs.Length;
            res.MAE = sumAbs / inputs.Length;
        }

        static void PrintResult(string tag, TrainResult r)
        {
            string status = r.Converged ? "OK" : "LIMIT";
            Console.WriteLine($"[{tag}] seed={r.Seed,4} | эпох={r.Epochs,7} | " +
                              $"Es={r.FinalError,12:F6} | acc={r.Accuracy:F2} | " +
                              $"MAE={r.MAE:F4} | {status}");
        }

        static void PrintSummaryTable(string title, List<TrainResult> results)
        {
            Console.WriteLine($"\n{title}");
            Console.WriteLine($"{"Seed",6} | {"Эпохи",8} | {"Итог.Es",14} | " +
                              $"{"Accuracy",9} | {"MAE",8}");
            foreach (var r in results)
                Console.WriteLine($"{r.Seed,6} | {r.Epochs,8} | {r.FinalError,14:F6} | " +
                                  $"{r.Accuracy,9:F4} | {r.MAE,8:F4}");
        }

        static void PrintStability(List<TrainResult> A, List<TrainResult> B)
        {
            Console.WriteLine("\nУстойчивость сходимости (5 запусков)");
            foreach (var pair in new[] { Tuple.Create("A (MSE)", A), Tuple.Create("B (BCE)", B) })
            {
                string name = pair.Item1;
                var list = pair.Item2;
                int conv = list.Count(r => r.Converged);
                var eps = list.Where(r => r.Converged).Select(r => r.Epochs).ToList();
                if (eps.Count > 0)
                    Console.WriteLine($"{name}: сошлось {conv}/{list.Count}, " +
                                      $"эпохи от {eps.Min()} до {eps.Max()}");
                else
                    Console.WriteLine($"{name}: сошлось 0/{list.Count}");
            }
        }

        static void PrintResults(NeuralNetwork net, double[][] inputs,
            double[][] rawInputs, double[] rawTargets, string mode)
        {
            Console.WriteLine($"{"A",6} {"B",6} {"Ожид.",7} {"ŷ_norm",9} {"ŷ_raw",10} {"Класс",7} {"Вердикт",9}");
            for (int i = 0; i < inputs.Length; i++)
            {
                double output = net.Forward(inputs[i]);
                double denormOutput = DenormalizeTarget(output);

                double dist0 = Math.Abs(denormOutput - C0);
                double distC1 = Math.Abs(denormOutput - C1);
                double predClass = dist0 < distC1 ? C0 : C1;
                bool ok = Math.Abs(predClass - rawTargets[i]) < 1e-9;
                string verdict = ok ? "Верно" : "Неверно";

                Console.WriteLine($"{rawInputs[i][0],6} {rawInputs[i][1],6} {rawTargets[i],7} " +
                                  $"{output,9:F4} {denormOutput,10:F4} {predClass,7:F0} " +
                                  $"{verdict,9}");
            }
        }

        static double HeuristicExpected(double a, double b)
        {
            int classA = Math.Abs(a - C0) < Math.Abs(a - C1) ? 0 : 1;
            int classB = Math.Abs(b - C0) < Math.Abs(b - C1) ? 0 : 1;
            int xor = classA ^ classB;
            return xor == 0 ? C0 : C1;
        }

        static double NearestClass(double yRaw)
            => Math.Abs(yRaw - C0) < Math.Abs(yRaw - C1) ? C0 : C1;

        static void RunInteractiveMode(NeuralNetwork net)
        {
            Console.WriteLine("\nРЕЖИМ ФУНКЦИОНИРОВАНИЯ СЕТИ (представительная, конфигурация Б)");
            Console.WriteLine("Введите пару A B из [-10;10] через пробел, 'q' — выход.");

            Console.WriteLine("\n4 примера из обучающей выборки (эталон из условия)");
            int okTrain = 0;
            for (int i = 0; i < rawInputs.Length; i++)
                if (PrintPrediction(net, rawInputs[i][0], rawInputs[i][1], rawTargets[i])) okTrain++;
            Console.WriteLine($"Итог по обучающей выборке: {okTrain}/{rawInputs.Length}");

            Console.WriteLine("\nДополнительные точки вне обучающей выборки");
            Console.WriteLine("Эталон рассчитан по знаковой аналогии с XOR:");
            Console.WriteLine("  класс(A) = (A ближе к c1) ? 1 : 0;");
            Console.WriteLine("  класс(B) = (B ближе к c1) ? 1 : 0;");
            Console.WriteLine("  ожидаемый = класс(A) XOR класс(B)  ->  0 -> c0, 1 -> c1");
            Console.WriteLine();

            double[][] extra = new double[][]
            {
                new double[] {  5,  5 },
                new double[] { -3,  7 },
                new double[] {  2, -8 },
                new double[] { -9, -2 }
            };

            int okExtra = 0;
            foreach (var p in extra)
            {
                double expected = HeuristicExpected(p[0], p[1]);
                if (PrintPrediction(net, p[0], p[1], expected)) okExtra++;
            }
            Console.WriteLine($"Итог по дополнительным точкам: {okExtra}/{extra.Length}");

            Console.WriteLine("\nИнтерактивный режим");
            while (true)
            {
                Console.Write("A B > ");
                string line = Console.ReadLine();
                if (line == null || line.Trim().ToLower() == "q") break;

                string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
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
                    PrintPrediction(net, a, b, expected);
                }
            }
        }

        static bool PrintPrediction(NeuralNetwork net, double a, double b, double? expected)
        {
            double[] testInput = new double[] { Normalize(a), Normalize(b) };
            double yNorm = net.Forward(testInput);
            double yRaw = DenormalizeTarget(yNorm);
            double cls = NearestClass(yRaw);

            double dist0 = Math.Abs(yRaw - C0);
            double distC1 = Math.Abs(yRaw - C1);
            double nearestVal = dist0 < distC1 ? C0 : C1;

            string expStr = expected.HasValue ? expected.Value.ToString("F0") : "—";
            bool ok = true;
            string verdict = "";
            if (expected.HasValue)
            {
                ok = Math.Abs(cls - expected.Value) < 1e-9;
                verdict = ok ? "Верно" : "Неверно";
            }

            Console.WriteLine($"A={a,5}, B={b,5} -> ŷ_norm={yNorm:F4}, " +
                              $"ŷ_raw={yRaw,8:F4}, класс={cls,4:F0} " +
                              $"(ближе к {nearestVal,4:F0}), " +
                              $"ожидалось={expStr,5} {verdict}");

            return ok;
        }

        static string GetProjectDir()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string dir = Path.GetFullPath(Path.Combine(baseDir, @"..\..\..\"));
            return dir;
        }

        static void GeneratePlots(List<TrainResult> resultsA, List<TrainResult> resultsB,
            TrainResult repA, TrainResult repB)
        {
            string dir = GetProjectDir();

            var plotConv = new Plot();
            plotConv.Title("Сходимость обучения: MSE vs BCE");
            plotConv.XLabel("Номер эпохи");
            plotConv.YLabel("Суммарная ошибка Es");

            double[] epA = Enumerable.Range(0, repA.History.Count).Select(i => (double)i).ToArray();
            double[] epB = Enumerable.Range(0, repB.History.Count).Select(i => (double)i).ToArray();

            var scA = plotConv.Add.Scatter(epA, repA.History.ToArray());
            scA.LegendText = "Конфигурация А (MSE)";
            scA.Color = Colors.Blue;
            scA.LineWidth = 2;

            var scB = plotConv.Add.Scatter(epB, repB.History.ToArray());
            scB.LegendText = "Конфигурация Б (BCE)";
            scB.Color = Colors.Red;
            scB.LineWidth = 2;

            var lineEeMSE = plotConv.Add.HorizontalLine(EeMSE);
            lineEeMSE.Color = Colors.Blue;
            lineEeMSE.LinePattern = LinePattern.Dashed;
            lineEeMSE.LegendText = $"Ee MSE = {EeMSE}";

            var lineEeBCE = plotConv.Add.HorizontalLine(EeBCE);
            lineEeBCE.Color = Colors.Red;
            lineEeBCE.LinePattern = LinePattern.Dotted;
            lineEeBCE.LegendText = $"Ee BCE = {EeBCE}";

            plotConv.ShowLegend();
            plotConv.SavePng(Path.Combine(dir, "plot_convergence.png"), 900, 600);

            var plotEpochs = new Plot();
            plotEpochs.Title("Число эпох до сходимости по 5 запускам");
            plotEpochs.XLabel("Seed");
            plotEpochs.YLabel("Эпохи");

            double[] seeds = resultsA.Select(r => (double)r.Seed).ToArray();
            double[] epAD = resultsA.Select(r => r.Converged ? (double)r.Epochs : safetyLimit).ToArray();
            double[] epBD = resultsB.Select(r => r.Converged ? (double)r.Epochs : safetyLimit).ToArray();

            var barsA = plotEpochs.Add.Bars(seeds.Select(s => s - 1.5).ToArray(), epAD);
            barsA.LegendText = "Конфигурация А (MSE)";
            barsA.Color = Colors.Blue;

            var barsB = plotEpochs.Add.Bars(seeds.Select(s => s + 1.5).ToArray(), epBD);
            barsB.LegendText = "Конфигурация Б (BCE)";
            barsB.Color = Colors.Red;

            plotEpochs.ShowLegend();
            plotEpochs.SavePng(Path.Combine(dir, "plot_epochs.png"), 900, 600);

            var multiplot = new Multiplot();
            multiplot.AddPlots(2);

            Plot plotSurfA = multiplot.GetPlot(0);
            Plot plotSurfB = multiplot.GetPlot(1);

            GenerateSurfacePlot(plotSurfA, repA.Net, "Конфигурация А (MSE)");
            GenerateSurfacePlot(plotSurfB, repB.Net, "Конфигурация Б (BCE)");

            multiplot.SavePng(Path.Combine(dir, "plot_surfaces.png"), 1400, 600);

            var plotMae = new Plot();
            plotMae.Title("Сравнение точности восстановления шкалы [c0;c1]");
            plotMae.XLabel("Seed");
            plotMae.YLabel("Средняя абсолютная ошибка (MAE)");

            double[] maeAD = resultsA.Select(r => r.MAE).ToArray();
            double[] maeBD = resultsB.Select(r => r.MAE).ToArray();

            var barsMaeA = plotMae.Add.Bars(seeds.Select(s => s - 1.5).ToArray(), maeAD);
            barsMaeA.LegendText = "Конфигурация А (MSE)";
            barsMaeA.Color = Colors.Blue;

            var barsMaeB = plotMae.Add.Bars(seeds.Select(s => s + 1.5).ToArray(), maeBD);
            barsMaeB.LegendText = "Конфигурация Б (BCE)";
            barsMaeB.Color = Colors.Red;

            plotMae.ShowLegend();
            plotMae.SavePng(Path.Combine(dir, "plot_mae.png"), 900, 600);

            Console.WriteLine("\nСохранены графики в папку проекта:");
            Console.WriteLine("  plot_convergence.png");
            Console.WriteLine("  plot_epochs.png");
            Console.WriteLine("  plot_surfaces.png");
            Console.WriteLine("  plot_mae.png");
        }

        static void GenerateSurfacePlot(Plot plot, NeuralNetwork net, string title)
        {
            int gridSize = 60;
            double[,] zData = new double[gridSize, gridSize];

            for (int i = 0; i < gridSize; i++)
            {
                for (int j = 0; j < gridSize; j++)
                {
                    double a = -10 + 20.0 * i / (gridSize - 1);
                    double b = -10 + 20.0 * j / (gridSize - 1);
                    double[] inp = { Normalize(a), Normalize(b) };
                    double yRaw = DenormalizeTarget(net.Forward(inp));
                    zData[j, i] = yRaw;
                }
            }

            var heatmap = plot.Add.Heatmap(zData);
            heatmap.Extent = new CoordinateRect(-10, 10, -10, 10);
            heatmap.FlipVertically = true;
            heatmap.Colormap = new ScottPlot.Colormaps.Turbo();

            for (int k = 0; k < rawInputs.Length; k++)
            {
                double x = rawInputs[k][0];
                double y = rawInputs[k][1];
                double val = rawTargets[k];
                var marker = plot.Add.Marker(x, y);
                marker.Color = val == C0 ? Colors.Blue : Colors.Red;
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