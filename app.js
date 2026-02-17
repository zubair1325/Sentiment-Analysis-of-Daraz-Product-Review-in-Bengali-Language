const express = require("express");
const app = express();
const path = require("path");
const mongoose = require("mongoose");
const ejsMate = require("ejs-mate");
const methodOverride = require("method-override");
const multer = require("multer");
const { spawn } = require("child_process");
const csv = require("csv-parser");
const ExpressError = require("./utils/ExpressError");
const asyncAwait = require("./utils/asyncAwait");
const upload = multer({ storage: multer.memoryStorage() });
const fs = require("fs");
const xlsx = require("xlsx");
const session = require("express-session");
const { log } = require("console");

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));
app.use(express.static(path.join(__dirname, "public")));
app.use(express.urlencoded({ extended: true }));
app.use(methodOverride("_method"));
app.engine("ejs", ejsMate);
app.use(express.json());

app.get("/home", (req, res) => {
  res.render("./pages/landing");
});
app.use(
  session({
    secret: "your_secret_key",
    resave: false,
    saveUninitialized: true,
  })
);

app.post("/home", upload.single("datafile"), (req, res) => {
  const selectedAlgos = req.body.algorithms;
  const algoList = selectedAlgos.split(",").map(Number);

  if (!req.file) return res.status(400).send("No file uploaded.");

  const py = spawn("python", [
    path.join(__dirname, "pythonCodeFolder", "main.py"),
  ]);

  py.stdin.write(req.file.buffer);
  py.stdin.end();

  let rawCSV = "";
  let error = "";

  py.stdout.on("data", (data) => {
    rawCSV += data.toString();
  });

  py.stderr.on("data", (data) => {
    error += data.toString();
  });

  py.on("close", (code) => {
    if (code !== 0) {
      console.error("Python Error:", error);
      return res.status(500).send("Python preprocessing failed.");
    }

    const results = rawCSV
      .trim()
      .split("\n")
      .slice(1)
      .map((line) => line.split(","));

    const inputTexts = results.map((r) => r[0]);

    const processedPath = path.join(__dirname, "pythonCodeFolder", "processed");
    const labelSheet = xlsx.utils.sheet_to_json(
      xlsx.readFile(path.join(processedPath, "label_encoded.xlsx")).Sheets[
        "Sheet1"
      ]
    );
    const vectorSheet = xlsx.utils.sheet_to_json(
      xlsx.readFile(path.join(processedPath, "Vectorization.xlsx")).Sheets[
        "Sheet1"
      ]
    );
    const tokenSheet = xlsx.utils.sheet_to_json(
      xlsx.readFile(path.join(processedPath, "Tokenization.xlsx")).Sheets[
        "Sheet1"
      ]
    );

    const labels = labelSheet.map((r) => r.encoded_label);
    const tfidfValues = vectorSheet.map((row) => Object.values(row));
    const featureNames = Object.keys(vectorSheet[0]);

    const algoPy = spawn("python", [
      path.join(__dirname, "pythonCodeFolder", "ml_models.py"),
    ]);

    const pyInput = JSON.stringify({
      inputTexts,
      tfidfValues,
      tokenizedData: tokenSheet,
      labels,
      algoList,
      featureNames,
      requestFeatureImportance: true,
      requestChi2: true,
    });

    algoPy.stdin.write(pyInput);
    algoPy.stdin.end();

    let rawJSON = "";
    let mlError = "";

    algoPy.stdout.on("data", (data) => {
      rawJSON += data.toString();
    });

    algoPy.stderr.on("data", (data) => {
      mlError += data.toString();
    });

    algoPy.on("close", (code) => {
      if (code !== 0) {
        console.error("ML Python Error:", mlError);
        return res.status(500).send("ML Model processing failed.");
      }

      const output = JSON.parse(rawJSON);
      const tableData = output.table;
      const summary = output.summary;
      const featureImportance = output.featureImportance || null;
      const chi2Features = output.chi2Features || null;

      req.session.summary = summary;
      req.session.featureImportance = featureImportance;
      req.session.chi2Features = chi2Features;

      const excelData = [Object.keys(tableData[0])].concat(
        tableData.map((row) => Object.values(row))
      );
      const worksheet = xlsx.utils.aoa_to_sheet(excelData);
      const workbook = xlsx.utils.book_new();
      xlsx.utils.book_append_sheet(workbook, worksheet, "Results");

      const outputPath = path.join(processedPath, "ml_predictions.xlsx");
      xlsx.writeFile(workbook, outputPath);

      res.render("./pages/done", {
        downloadPath: "/download",
        summaryAvailable: true,
        featureImportanceAvailable: !!featureImportance,
        chi2FeatureAvailable: !!chi2Features,
      });
    });
  });
});

app.get("/metrics", (req, res) => {
  if (!req.session || !req.session.summary) {
    return res
      .status(404)
      .json({ status: "error", error: "No metrics found. Run /home first." });
  }
  res.json({ status: "success", metrics: req.session.summary });
});
app.get("/metrics-view", (req, res) => {
  res.render("pages/lol");
});

app.get("/chi2-features", (req, res) => {
  const chi2Features = req.session.chi2Features || {};
  res.render("pages/chi2Features", { chi2Features });
});

app.get("/feature-importance", (req, res) => {
  const featureImportance = req.session.featureImportance || {};
  res.render("./pages/featureImportance", { featureImportance });
});

app.get("/download", (req, res) => {
  const outputPath = path.join(
    __dirname,
    "pythonCodeFolder",
    "processed",
    "ml_predictions.xlsx"
  );
  res.download(outputPath, "ml_predictions.xlsx", (err) => {
    if (err) console.error("Download error:", err);
  });
});

app.get("/output", (req, res) => {
  const summary = req.session.summary;
  if (!summary) return res.status(400).send("No summary found.");

  const chartData = {};

  for (const [model, stats] of Object.entries(summary)) {
    const counts = stats?.counts || {};
    const total = Object.values(counts).reduce(
      (sum, v) => sum + (typeof v === "number" ? v : 0),
      0
    );

    chartData[model] = {
      Neutral: total ? (((counts[2] || 0) / total) * 100).toFixed(1) : "0.0",
      Negative: total ? (((counts[1] || 0) / total) * 100).toFixed(1) : "0.0",
      Positive: total ? (((counts[0] || 0) / total) * 100).toFixed(1) : "0.0",
    };
  }

  res.render("pages/output", { chartData });
});

app.get("/us", (req, res) => {
  res.render("./pages/aboutUs");
});
app.get("/feedback", (req, res) => {
  res.render("./pages/feedback");
});
app.post("/feedback", (req, res) => {
  res.send("Feed back send");
});
app.get("/donate", (req, res) => {
  res.render("./pages/donateUs");
});
app.post("/donate", (req, res) => {
  res.send("Payment Success");
});

app.use((err, req, res, next) => {
  let { status = 410, message = "something went wrong on the backed" } = err;
  res.status(status).send(message);
});
app.listen(8080, () => {
  console.log("server online ");
});
