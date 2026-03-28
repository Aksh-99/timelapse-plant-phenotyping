import React, { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";

function mapStage(stageLabel) {
  if (stageLabel === 0) return "SEED";
  if (stageLabel === 1) return "GERMINATING";
  if (stageLabel === 2) return "SPROUTED";
  return "UNKNOWN";
}

function mapSprite(stageLabel, mood) {
  // CHILLY STATE
  if (mood === "Chilly and sluggish") {
    if (stageLabel === 0) return "/seed-chilly.png";
    if (stageLabel === 1) return "/germination-chilly.png";
    if (stageLabel === 2) return "/sprout-chilly.png";
  }

  // RESTING STATE
  if (mood === "Resting") {
    if (stageLabel === 1) return "/germination-resting.png";
    if (stageLabel === 2) return "/sprout-resting.png";
  }

  // NORMAL STATES
  if (stageLabel === 0) return "/seed-sleeping.png";
  if (stageLabel === 1) return "/seed-growing.png";
  if (stageLabel === 2) return "/seed-sprouted.png";

  return "/seed-sleeping.png";
}

function getPredictedTomorrow(currentHeight, nextHeight) {
  if (Number.isFinite(nextHeight)) return nextHeight;
  return currentHeight;
}

function getGrowthDelta(previousHeight, currentHeight) {
  return currentHeight - previousHeight;
}

function getMood(stage, growthDelta, temperature) {
  if (temperature <= 40 && growthDelta <= 0.1) return "Chilly and sluggish";

  if (stage === "SEED") {
    if (growthDelta <= 0) return "Dreaming...";
    if (growthDelta <= 0.2) return "Waking up...";
    return "Stirring...";
  }

  if (stage === "GERMINATING") {
    if (growthDelta <= 0) return "Resting";
    if (growthDelta <= 0.3) return "Waking up...";
    if (growthDelta <= 0.6) return "Growing!";
    return "Thriving!";
  }

  if (stage === "SPROUTED") {
    if (growthDelta <= 0) return "Resting";
    if (growthDelta <= 0.2) return "Growing...";
    if (growthDelta <= 0.5) return "Thriving!";
    return "Surging!";
  }

  return "Resting";
}

function getVelocityBand(heightToday, predictedTomorrow) {
  const safeHeight = Math.max(heightToday, 0.1);
  const velocity = (predictedTomorrow - heightToday) / safeHeight;

  if (velocity <= 0) return "Sleeping";
  if (velocity <= 0.1) return "Resting";
  if (velocity <= 0.3) return "Growing";
  if (velocity <= 0.6) return "Thriving";
  return "Surging";
}

function getVelocityFill(band) {
  if (band === "Sleeping") return 1;
  if (band === "Resting") return 2;
  if (band === "Growing") return 3;
  if (band === "Thriving") return 4;
  if (band === "Surging") return 5;
  return 1;
}

function velocityColor(band) {
  if (band === "Sleeping") return "#6b7280";
  if (band === "Resting") return "#84cc16";
  if (band === "Growing") return "#4ade80";
  if (band === "Thriving") return "#22c55e";
  if (band === "Surging") return "#00ff66";
  return "#243744";
}

function getStageText(stage, mood) {
  if (stage === "SEED") return mood;
  if (stage === "GERMINATING") return mood;
  if (stage === "SPROUTED") return mood;
  return "...";
}

function PixelBar({ filled, total = 10, small = false }) {
  return (
    <div style={{ display: "flex", gap: 2, alignItems: "center" }}>
      {Array.from({ length: total }).map((_, i) => (
        <div
          key={i}
          style={{
            width: small ? 8 : 5,
            height: small ? 10 : 7,
            background: i < filled ? "#4a8a30" : "#1e2a18",
            border: "1px solid #2a5a20",
            flexShrink: 0,
          }}
        />
      ))}
    </div>
  );
}

function PetSprite({ data }) {
  const animationName =
    data.stage === "SEED"
      ? "bounce"
      : data.stage === "GERMINATING"
      ? "pulse"
      : "sway";

  const soilBottom = "10%";
  const seedLift = "50px";

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "clamp(300px, 55vh, 520px)",
        margin: "0 auto",
        overflow: "hidden",
      }}
    >
      <img
        src="/soil.png"
        alt="soil"
        style={{
          position: "absolute",
          bottom: soilBottom,
          left: "50%",
          transform: "translateX(-50%)",
          width: "clamp(500px, 80%, 420px)",
          imageRendering: "pixelated",
          zIndex: 1,
        }}
      />

      <img
        src={data.sprite}
        alt={data.stage}
        style={{
          position: "absolute",
          bottom: `calc(${soilBottom} + ${seedLift})`,
          left: "50%",
          transform: "translateX(-50%)",
          width: "clamp(430px, 30%, 170px)",
          imageRendering: "pixelated",
          animation: `${animationName} 2s ease-in-out infinite`,
          zIndex: 2,
          filter: "drop-shadow(0 6px 10px rgba(0,0,0,0.6))",
        }}
      />

      <div
        style={{
          position: "absolute",
          bottom: "80%",
          width: "100%",
          textAlign: "center",
          fontSize: "clamp(23px, 1.7vw, 14px)",
          color: "#d4e880",
          letterSpacing: 1,
          animation: "blink 1.4s step-end infinite",
          padding: "0 8px",
        }}
      >
        {data.stageText}
      </div>
    </div>
  );
}

function StatChip({ label, value, children }) {
  return (
    <div
      style={{
        border: "2px solid #2a4a30",
        background: "#0d1a1f",
        color: "#d4e880",
        padding: "10px 12px",
        borderRadius: 4,
        minWidth: 140,
        flex: "1 1 140px",
      }}
    >
      <div
        style={{
          fontSize: "clamp(8px, 1vw, 12px)",
          color: "#7fb063",
          marginBottom: 8,
          letterSpacing: 1,
        }}
      >
        {label}
      </div>
      {children ? (
        <div>{children}</div>
      ) : (
        <div style={{ fontSize: 9, lineHeight: 1.6 }}>{value}</div>
      )}
    </div>
  );
}

function ForecastCard({ selected, avgMAE }) {
  return (
    <div className="panel">
      <div className="panel-title">FORECAST CARD</div>

      <div
        style={{
          border: "2px solid #39586c",
          background: "linear-gradient(180deg, #102331 0%, #0d1a1f 100%)",
          borderRadius: 8,
          padding: 16,
        }}
      >
        <div
          style={{
            fontSize: "clamp(10px, 1.2vw, 12px)",
            color: "#93b17d",
            marginBottom: 14,
          }}
        >
          REAL-TIME HEIGHT FORECAST
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div style={{ fontSize: "clamp(14px, 1.8vw, 20px)", color: "#d4e880" }}>
            {selected.height.toFixed(1)} in
          </div>

          <div style={{ fontSize: 14, color: "#6aa7d8" }}>→</div>

          <div style={{ fontSize: "clamp(14px, 1.8vw, 20px)", color: "#7fc7ff" }}>
            {selected.predictedTomorrow.toFixed(1)} in
          </div>

          <div
            style={{
              padding: "6px 10px",
              border: "2px solid #2a4050",
              borderRadius: 4,
              background: "#132432",
              fontSize: 8,
              color: "#c8aa60",
            }}
          >
            ± {avgMAE.toFixed(2)} in MAE
          </div>
        </div>

        <div
          style={{
            marginTop: 14,
            fontSize: 8,
            color: "#9cb7c6",
            lineHeight: 1.8,
          }}
        >
          Today: Day {selected.day} actual measured height. Tomorrow: model-driven
          forecast using your time-series growth pattern.
        </div>
      </div>
    </div>
  );
}

function ChartCard({ days, current, avgMAE }) {
  const chartData = days.map((d) => {
    const actual = Number(d.height);
    const predicted = Number(d.predictedTomorrow);
    const mae = Number(avgMAE);

    return {
      day: Number(d.day),
      actual: Number.isFinite(actual) ? actual : null,
      predicted: Number.isFinite(predicted) ? predicted : null,
      upper:
        Number.isFinite(predicted) && Number.isFinite(mae)
          ? predicted + mae
          : null,
      lower:
        Number.isFinite(predicted) && Number.isFinite(mae)
          ? Math.max(0, predicted - mae)
          : null,
    };
  });

  const currentDay = Number(days[current]?.day);

  return (
    <div className="panel">
      <div className="panel-title">PREDICTED VS ACTUAL HEIGHT</div>

      <div
        style={{
          height: 330,
          border: "2px solid #2a4050",
          borderRadius: 8,
          background: "#0b141c",
          padding: 10,
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 15, right: 20, left: 0, bottom: 10 }}
          >
            <CartesianGrid stroke="#243744" strokeDasharray="3 3" />

            <XAxis
              dataKey="day"
              stroke="#7fb063"
              tick={{ fontSize: 10, fill: "#93b17d" }}
              tickLine={false}
              axisLine={{ stroke: "#2a4050" }}
              label={{
                value: "Day",
                position: "insideBottom",
                offset: -5,
                fill: "#7fb063",
                fontSize: 10,
              }}
            />

            <YAxis
              stroke="#7fb063"
              tick={{ fontSize: 10, fill: "#93b17d" }}
              tickLine={false}
              axisLine={{ stroke: "#2a4050" }}
              label={{
                value: "Height (in)",
                angle: -90,
                position: "insideLeft",
                fill: "#7fb063",
                fontSize: 10,
              }}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: "#0d1a1f",
                border: "2px solid #2a4050",
                borderRadius: 6,
                fontSize: 10,
                color: "#d4e880",
              }}
              labelStyle={{ color: "#c8aa60" }}
            />

            <Legend wrapperStyle={{ fontSize: 10, color: "#d4e880" }} />

            <Line
              type="linear"
              dataKey="upper"
              stroke="#1f4b68"
              strokeWidth={1}
              dot={false}
              name="Upper Error Bound"
              connectNulls
              isAnimationActive={false}
            />

            <Line
              type="linear"
              dataKey="lower"
              stroke="#1f4b68"
              strokeWidth={1}
              dot={false}
              name="Lower Error Bound"
              connectNulls
              isAnimationActive={false}
            />

            <Line
              type="linear"
              dataKey="predicted"
              stroke="#60a5fa"
              strokeWidth={3}
              strokeDasharray="6 4"
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
              name="Predicted Height"
              connectNulls
              isAnimationActive={false}
            />

            <Line
              type="linear"
              dataKey="actual"
              stroke="#4ade80"
              strokeWidth={3}
              dot={{ r: 5 }}
              activeDot={{ r: 7 }}
              name="Actual Height"
              connectNulls
              isAnimationActive={false}
            />

            {Number.isFinite(currentDay) && (
              <ReferenceLine
                x={currentDay}
                stroke="#c8aa60"
                strokeDasharray="4 4"
                label={{
                  value: "Selected Day",
                  fill: "#c8aa60",
                  fontSize: 10,
                  position: "top",
                }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

const circleBtnStyle = {
  width: 26,
  height: 26,
  borderRadius: "50%",
  background: "#162030",
  border: "2px solid #2a4050",
  cursor: "pointer",
  flexShrink: 0,
};

export default function PlantGrowthDashboard() {
  const [days, setDays] = useState([]);
  const [current, setCurrent] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/height_dataset.csv")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Could not load CSV. HTTP ${res.status}`);
        }
        return res.text();
      })
      .then((csvText) => {
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const rawRows = results.data
              .map((row) => {
                const day = Number(
                  row.day ?? row.Day ?? row.DAY ?? row[" day "] ?? row["Day "]
                );

                const stageLabel = Number(
                  row.stage_label ??
                    row.stage ??
                    row.Stage ??
                    row.stageLabel ??
                    row["stage label"]
                );

                const temperature = Number(
                  row.temperature ??
                    row.temp ??
                    row.Temp ??
                    row.Temperature ??
                    row["temperature_f"] ??
                    row["Temperature (F)"]
                );

                const height = Number(
                  row.height ??
                    row.Height ??
                    row["height_in"] ??
                    row["Height(in)"] ??
                    row["Height (in)"] ??
                    row["plant_height"]
                );

                return {
                  day,
                  stageLabel,
                  temperature: Number.isFinite(temperature) ? temperature : 0,
                  height: Number.isFinite(height) ? height : 0,
                };
              })
              .filter((d) => Number.isFinite(d.day))
              .sort((a, b) => a.day - b.day);

            const parsed = rawRows.map((row, index) => {
              const stage = mapStage(row.stageLabel);

              const prevHeight =
                index > 0 ? rawRows[index - 1].height : row.height;

              const nextHeight =
                index < rawRows.length - 1 ? rawRows[index + 1].height : row.height;

              const predictedTomorrow = getPredictedTomorrow(row.height, nextHeight);

              const growthDelta = getGrowthDelta(prevHeight, row.height);

              const mood = getMood(stage, growthDelta, row.temperature);

              const velocityBand = getVelocityBand(row.height, predictedTomorrow);

              return {
                day: row.day,
                label: `Day ${row.day}`,
                stage,
                sprite: mapSprite(row.stageLabel, mood),
                hp: Math.max(1, Math.min(10, 8 + (row.stageLabel || 0))),
                water: Math.max(1, Math.min(10, 7 + (row.stageLabel || 0))),
                isNight: false,
                temp: row.temperature,
                height: row.height,
                predictedTomorrow,
                growthDelta,
                mood,
                velocityBand,
                stageText: getStageText(stage, mood),
              };
            });

            if (!parsed.length) {
              throw new Error(
                "CSV loaded, but no valid rows were parsed. Check your CSV column names."
              );
            }

            setDays(parsed);
            setCurrent(0);
            setLoading(false);
          },
          error: (err) => {
            setError(`CSV parse error: ${err.message}`);
            setLoading(false);
          },
        });
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const avgHeight = useMemo(() => {
    if (!days.length) return "0.00";
    const total = days.reduce((sum, d) => sum + d.height, 0);
    return (total / days.length).toFixed(2);
  }, [days]);

  const avgMAE = useMemo(() => {
    if (!days.length) return 0;
    return (
      days.reduce((sum, d) => sum + Math.abs(d.growthDelta), 0) / days.length
    );
  }, [days]);

  const selected = days.length ? days[current] : null;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

        * { box-sizing: border-box; }

        html, body, #root {
          margin: 0;
          width: 100%;
          min-height: 100%;
        }

        body {
          background:
            radial-gradient(circle at top, rgba(44,80,90,0.35), transparent 30%),
            linear-gradient(180deg, #071017 0%, #0d1720 100%);
          overflow-x: hidden;
        }

        button {
          font-family: "Press Start 2P", monospace;
        }

        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }

        @keyframes bounce {
          0%, 100% { transform: translateX(-50%) translateY(0); }
          50% { transform: translateX(-50%) translateY(-6px); }
        }

        @keyframes sway {
          0%, 100% { transform: translateX(-50%) rotate(-4deg); }
          50% { transform: translateX(-50%) rotate(4deg); }
        }

        @keyframes pulse {
          0%, 100% { transform: translateX(-50%) scale(1); }
          50% { transform: translateX(-50%) scale(1.08); }
        }

        .dashboard-grid {
          display: grid;
          grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
          gap: 24px;
          align-items: start;
          width: min(1320px, 100%);
          margin: 0 auto;
        }

        .device-shell {
          position: sticky;
          top: 20px;
          width: min(100%, 380px);
          margin: 28px auto 0;
        }

        .device-body {
          position: relative;
          width: 100%;
          aspect-ratio: 0.75 / 1.08;
          background: #0d1a1f;
          border-radius: 50% / 40%;
          border: 4px solid #1e3040;
          padding: 8.5% 8.5% 22%;
          box-shadow: inset 0 0 30px #000, 0 8px 32px rgba(0,0,0,0.7);
        }

        .screen-frame {
          background: #0a1520;
          border: 3px solid #2a4050;
          border-radius: 8px;
          padding: 4px;
          height: 100%;
        }

        .screen-inner {
          background: #2a3520;
          border: 2px solid #3a5030;
          border-radius: 4px;
          padding: 12px;
          position: relative;
          height: 100%;
          overflow: hidden;
        }

        .screen-grid {
          position: absolute;
          inset: 0;
          background-image:
            linear-gradient(rgba(80,120,60,0.08) 1px, transparent 1px),
            linear-gradient(90deg, rgba(80,120,60,0.08) 1px, transparent 1px);
          background-size: 12px 12px;
          pointer-events: none;
        }

        .device-controls {
          position: absolute;
          bottom: 4.8%;
          left: 50%;
          transform: translateX(-50%);
          display: flex;
          gap: 12px;
          align-items: center;
          width: max-content;
        }

        .info-col {
          display: flex;
          flex-direction: column;
          gap: 18px;
          min-width: 0;
          padding-top: 28px;
        }

        .panel {
          border: 3px solid #2a4050;
          background: rgba(10,21,32,0.92);
          border-radius: 8px;
          padding: 18px;
          box-shadow: 0 8px 32px rgba(0,0,0,0.2);
          min-width: 0;
        }

        .panel-title {
          font-size: 10px;
          color: #7fb063;
          margin-bottom: 14px;
        }

        .log-row {
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: 12px;
          align-items: center;
          border-radius: 6px;
          padding: 12px;
          font-size: 8px;
        }

        .log-row > div {
          min-width: 0;
          overflow-wrap: anywhere;
        }

        @media (max-width: 980px) {
          .dashboard-grid {
            grid-template-columns: 1fr;
          }

          .device-shell {
            position: relative;
            top: 0;
            margin-top: 18px;
          }

          .info-col {
            padding-top: 0;
          }
        }

        @media (max-width: 640px) {
          .log-row {
            grid-template-columns: 1fr 1fr;
          }
        }
      `}</style>

      <div
        style={{
          minHeight: "100dvh",
          fontFamily: '"Press Start 2P", monospace',
          color: "#d4e880",
          padding: "clamp(14px, 2vw, 28px)",
        }}
      >
        {loading && (
          <div
            style={{
              color: "#d4e880",
              textAlign: "center",
              marginTop: 80,
              fontSize: 14,
            }}
          >
            Loading plant data...
          </div>
        )}

        {!loading && error && (
          <div
            style={{
              color: "#ff8f8f",
              background: "rgba(0,0,0,0.35)",
              border: "2px solid #803030",
              padding: 16,
              maxWidth: 900,
              margin: "40px auto",
              lineHeight: 1.8,
              fontSize: 12,
            }}
          >
            <div style={{ marginBottom: 12 }}>App could not load the CSV.</div>
            <div>{error}</div>
            <div style={{ marginTop: 12, color: "#d4e880" }}>
              Check that <code>public/height_dataset.csv</code> exists and that you
              installed <code>papaparse</code> and <code>recharts</code>.
            </div>
          </div>
        )}

        {!loading && !error && selected && (
          <>
            <div style={{ textAlign: "center", marginBottom: 18 }}>
              <div
                style={{
                  color: "#7fb063",
                  fontSize: "clamp(8px, 1.1vw, 10px)",
                  marginBottom: 10,
                }}
              >
                PLANT PHENOTYPING PET
              </div>
              <h1
                style={{
                  margin: 0,
                  fontSize: "clamp(12px, 2.2vw, 22px)",
                  color: "#c8aa60",
                  textShadow: "2px 2px 0 #000",
                  lineHeight: 1.4,
                }}
              >
                PLANT GROWTH PREDICTION SYSTEM
              </h1>
              <div style={{ marginTop: 12, fontSize: 10, color: "#93b17d" }}>
                Loaded {days.length} day(s) from CSV
              </div>
            </div>

            <div className="dashboard-grid">
              <div className="device-shell">
                <span
                  style={{
                    position: "absolute",
                    top: -22,
                    left: "50%",
                    transform: "translateX(-50%)",
                    fontSize: "clamp(8px, 1.1vw, 11px)",
                    color: "#c8aa60",
                    letterSpacing: 2,
                    whiteSpace: "nowrap",
                    textShadow: "1px 1px 0 #000",
                    zIndex: 2,
                  }}
                >
                  TAMAGOTCHI
                </span>

                <div className="device-body">
                  <div className="screen-frame">
                    <div className="screen-inner">
                      <div className="screen-grid" />

                      <div style={{ position: "relative", zIndex: 1, height: "100%" }}>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            marginBottom: 10,
                            gap: 8,
                          }}
                        >
                          <span
                            style={{
                              fontSize: "clamp(6px, 1vw, 8px)",
                              letterSpacing: 1,
                            }}
                          >
                            DUMPLING
                          </span>
                          <span style={{ fontSize: "clamp(10px, 1.5vw, 12px)" }}>
                            🌰
                          </span>
                        </div>

                        <div
                          style={{
                            background: "#1e2a18",
                            border: "1px solid #3a5030",
                            borderRadius: 2,
                            padding: "6px 8px",
                            display: "flex",
                            alignItems: "center",
                            gap: 6,
                            marginBottom: 12,
                            flexWrap: "nowrap",
                            overflow: "hidden",
                          }}
                        >
                          <span style={{ color: "#e05060", fontSize: 8, flexShrink: 0 }}>
                            ♥
                          </span>

                          <div style={{ minWidth: 0 }}>
                            <PixelBar filled={selected.hp} />
                          </div>

                          <span
                            style={{
                              color: selected.isNight ? "#7080d0" : "#e8b030",
                              fontSize: 8,
                              flexShrink: 0,
                            }}
                          >
                            {selected.isNight ? "☽" : "✦"}
                          </span>

                          <span
                            style={{ color: "#40a0e0", fontSize: 8, flexShrink: 0 }}
                          >
                            💧
                          </span>

                          <div style={{ minWidth: 0 }}>
                            <PixelBar filled={selected.water} />
                          </div>

                          <span
                            style={{
                              marginLeft: "auto",
                              fontSize: "clamp(6px, 0.9vw, 7px)",
                              whiteSpace: "nowrap",
                              flexShrink: 0,
                            }}
                          >
                            DAY {String(selected.day).padStart(2, "0")}
                          </span>
                        </div>

                        <PetSprite data={selected} />
                      </div>
                    </div>
                  </div>

                  <div className="device-controls">
                    <span style={{ fontSize: 7, color: "#4a6a50" }}>&lt;</span>

                    <button
                      onClick={() =>
                        setCurrent((prev) => (prev - 1 + days.length) % days.length)
                      }
                      style={circleBtnStyle}
                      aria-label="Previous day"
                    />

                    <button
                      onClick={() => setCurrent(0)}
                      style={circleBtnStyle}
                      aria-label="Reset to day 1"
                    />

                    <button
                      onClick={() => setCurrent((prev) => (prev + 1) % days.length)}
                      style={circleBtnStyle}
                      aria-label="Next day"
                    />

                    <span style={{ fontSize: 7, color: "#4a6a50" }}>&gt;</span>
                  </div>

                  <div
                    style={{
                      position: "absolute",
                      bottom: "6.2%",
                      right: "7%",
                      width: 10,
                      height: 10,
                      background: "#c8aa60",
                      transform: "rotate(45deg)",
                      opacity: 0.6,
                    }}
                  />
                </div>
              </div>

              <div className="info-col">
                <div className="panel">
                  <div className="panel-title">CURRENT PET STATUS</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
                    <StatChip label="Stage" value={selected.stage} />
                    <StatChip label="Height" value={`${selected.height.toFixed(1)} in`} />
                    <StatChip label="Temp" value={`${selected.temp.toFixed(2)} °F`} />
                    <StatChip label="Mood" value={selected.mood} />
                    <StatChip
                      label="Pred Tomorrow"
                      value={`${selected.predictedTomorrow.toFixed(1)} in`}
                    />
                    <StatChip
                      label="Growth Delta"
                      value={`${selected.growthDelta >= 0 ? "+" : ""}${selected.growthDelta.toFixed(2)} in`}
                    />
                    <StatChip label="Velocity Meter">
                      <div
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          justifyContent: "center",
                          gap: 10,
                          minHeight: 52,
                          textAlign: "center",
                          width: "100%",
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "center", width: "100%" }}>
                          <PixelBar
                            filled={getVelocityFill(selected.velocityBand)}
                            total={5}
                            small
                          />
                        </div>

                        <div
                          style={{
                            fontSize: "clamp(10px, 1.2vw, 14px)",
                            color: "#d4e880",
                            textAlign: "center",
                            width: "100%",
                            letterSpacing: 1,
                          }}
                        >
                          {selected.velocityBand}
                        </div>
                      </div>
                    </StatChip>
                  </div>
                </div>

                <ForecastCard selected={selected} avgMAE={avgMAE} />

                <ChartCard days={days} current={current} avgMAE={avgMAE} />

                <div className="panel">
                  <div className="panel-title">TIMELINE BUTTONS</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {days.map((d, i) => (
                      <button
                        key={d.day}
                        onClick={() => setCurrent(i)}
                        style={{
                          fontSize: 7,
                          padding: "8px 10px",
                          background: i === current ? "#1e3a20" : "#0d1a1f",
                          color: i === current ? "#d4e880" : "#6a9a50",
                          border: `2px solid ${
                            i === current ? "#6abf50" : "#2a4a30"
                          }`,
                          borderRadius: 4,
                          cursor: "pointer",
                          letterSpacing: 0.5,
                        }}
                      >
                        {d.label.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-title">GROWTH LOG</div>

                  <div
                    style={{
                      maxHeight: "420px",
                      overflowY: "auto",
                      paddingRight: "6px",
                    }}
                  >
                    <div
                      className="log-row"
                      style={{
                        border: "2px solid #2a4050",
                        background: "#0d1a1f",
                        fontSize: 7,
                        color: "#7fb063",
                        letterSpacing: 1,
                        marginBottom: 8,
                        position: "sticky",
                        top: 0,
                        zIndex: 2,
                      }}
                    >
                      <div>DAY</div>
                      <div style={{ textAlign: "center" }}>STAGE</div>
                      <div style={{ textAlign: "center" }}>MOOD</div>
                      <div style={{ textAlign: "center" }}>HEIGHT</div>
                      <div style={{ textAlign: "right" }}>TEMP</div>
                    </div>

                    <div style={{ display: "grid", gap: 10 }}>
                      {days.map((d, i) => (
                        <div
                          key={d.day}
                          className="log-row"
                          style={{
                            border: `2px solid ${i === current ? "#6abf50" : "#243744"}`,
                            borderLeft: `6px solid ${velocityColor(d.velocityBand)}`,
                            background:
                              i === current
                                ? "rgba(30,58,32,0.35)"
                                : "rgba(13,26,31,0.8)",
                          }}
                        >
                          <div>{d.label.toUpperCase()}</div>
                          <div style={{ textAlign: "center" }}>{d.stage.toUpperCase()}</div>
                          <div style={{ textAlign: "center" }}>{d.stageText.toUpperCase()}</div>
                          <div style={{ textAlign: "center" }}>{d.height.toFixed(1)} in</div>
                          <div style={{ textAlign: "right" }}>{d.temp.toFixed(2)} °F</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  <StatChip label="Avg Height" value={`${avgHeight} in`} />
                  <StatChip label="Avg MAE" value={`± ${avgMAE.toFixed(2)} in`} />
                  <StatChip label="Pet Name" value="DUMPLING" />
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}