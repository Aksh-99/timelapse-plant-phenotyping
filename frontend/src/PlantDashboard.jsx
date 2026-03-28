import React, { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts";
import { Sprout, Thermometer, Ruler, CalendarDays, Search, Sparkles, TrendingUp, Flower2 } from "lucide-react";
import { motion } from "framer-motion";

const plantData = [
  { day: 1, date: "Mar 12", temp: 41, stage: "Seed", height: 0.0 },
  { day: 2, date: "Mar 13", temp: 45, stage: "Seed", height: 0.0 },
  { day: 3, date: "Mar 14", temp: 43, stage: "Seed", height: 0.0 },
  { day: 4, date: "Mar 15", temp: 48, stage: "Germination", height: 0.2 },
  { day: 5, date: "Mar 16", temp: 50, stage: "Germination", height: 0.6 },
  { day: 6, date: "Mar 17", temp: 47, stage: "Germination", height: 1.0 },
  { day: 7, date: "Mar 18", temp: 44, stage: "Sprout", height: 1.4 },
  { day: 8, date: "Mar 19", temp: 46, stage: "Sprout", height: 1.9 },
  { day: 9, date: "Mar 20", temp: 49, stage: "Sprout", height: 2.5 },
  { day: 10, date: "Mar 21", temp: 53, stage: "Sprout", height: 3.1 },
  { day: 11, date: "Mar 22", temp: 52, stage: "Sprout", height: 3.8 },
  { day: 12, date: "Mar 23", temp: 50, stage: "Sprout", height: 4.2 },
];

const stageColors = {
  Seed: "bg-stone-100 text-stone-700 border-stone-200",
  Germination: "bg-amber-100 text-amber-700 border-amber-200",
  Sprout: "bg-emerald-100 text-emerald-700 border-emerald-200",
};

function TamagotchiPlant({ stage }) {
  if (stage === "Seed") {
    return (
      <motion.div
        initial={{ y: 0, scale: 1 }}
        animate={{ y: [0, -3, 0], scale: [1, 1.02, 1] }}
        transition={{ repeat: Infinity, duration: 2.2, ease: "easeInOut" }}
        className="relative mx-auto flex h-36 w-36 items-end justify-center"
      >
        <div className="absolute bottom-2 h-10 w-24 rounded-[40px] bg-amber-100" />
        <div className="absolute bottom-8 h-12 w-16 rounded-full bg-amber-700 shadow-inner" />
        <div className="absolute bottom-14 left-[46px] h-2 w-2 rounded-full bg-slate-900" />
        <div className="absolute bottom-14 right-[46px] h-2 w-2 rounded-full bg-slate-900" />
        <motion.div
          animate={{ rotate: [-8, 8, -8] }}
          transition={{ repeat: Infinity, duration: 1.8, ease: "easeInOut" }}
          className="absolute bottom-9 h-1 w-5 rounded-full bg-slate-900"
        />
      </motion.div>
    );
  }

  if (stage === "Germination") {
    return (
      <motion.div
        initial={{ y: 0 }}
        animate={{ y: [0, -4, 0] }}
        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
        className="relative mx-auto flex h-36 w-36 items-end justify-center"
      >
        <div className="absolute bottom-2 h-10 w-24 rounded-[40px] bg-amber-100" />
        <div className="absolute bottom-8 h-8 w-16 rounded-full bg-amber-700" />
        <motion.div
          animate={{ height: [30, 34, 30] }}
          transition={{ repeat: Infinity, duration: 1.8, ease: "easeInOut" }}
          className="absolute bottom-12 w-1 rounded-full bg-emerald-500"
        />
        <motion.div
          animate={{ rotate: [-12, -2, -12], scale: [1, 1.05, 1] }}
          transition={{ repeat: Infinity, duration: 1.8, ease: "easeInOut" }}
          className="absolute bottom-[38px] left-[63px] h-5 w-8 rounded-full bg-emerald-300"
          style={{ borderRadius: "100% 0 100% 0" }}
        />
        <motion.div
          animate={{ rotate: [12, 2, 12], scale: [1, 1.05, 1] }}
          transition={{ repeat: Infinity, duration: 1.8, ease: "easeInOut" }}
          className="absolute bottom-[38px] right-[63px] h-5 w-8 rounded-full bg-emerald-300"
          style={{ borderRadius: "0 100% 0 100%" }}
        />
        <div className="absolute bottom-16 left-[61px] h-2 w-2 rounded-full bg-slate-900" />
        <div className="absolute bottom-16 right-[61px] h-2 w-2 rounded-full bg-slate-900" />
        <div className="absolute bottom-[52px] h-1 w-4 rounded-full bg-slate-900" />
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ y: 0 }}
      animate={{ y: [0, -5, 0], rotate: [0, 1, 0, -1, 0] }}
      transition={{ repeat: Infinity, duration: 2.4, ease: "easeInOut" }}
      className="relative mx-auto flex h-36 w-36 items-end justify-center"
    >
      <div className="absolute bottom-2 h-10 w-24 rounded-[40px] bg-amber-100" />
      <div className="absolute bottom-8 h-8 w-16 rounded-full bg-amber-700" />
      <div className="absolute bottom-12 h-12 w-1 rounded-full bg-emerald-600" />
      <motion.div
        animate={{ rotate: [-10, 6, -10] }}
        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
        className="absolute bottom-[46px] left-[56px] h-8 w-10 bg-emerald-400"
        style={{ borderRadius: "100% 0 100% 0" }}
      />
      <motion.div
        animate={{ rotate: [10, -6, 10] }}
        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
        className="absolute bottom-[46px] right-[56px] h-8 w-10 bg-emerald-400"
        style={{ borderRadius: "0 100% 0 100%" }}
      />
      <motion.div
        animate={{ scale: [1, 1.06, 1] }}
        transition={{ repeat: Infinity, duration: 1.6, ease: "easeInOut" }}
        className="absolute bottom-[68px] flex h-12 w-12 items-center justify-center rounded-full bg-emerald-300 shadow"
      >
        <Flower2 className="h-7 w-7 text-emerald-700" />
      </motion.div>
      <div className="absolute bottom-[76px] left-[59px] h-2 w-2 rounded-full bg-slate-900" />
      <div className="absolute bottom-[76px] right-[59px] h-2 w-2 rounded-full bg-slate-900" />
      <div className="absolute bottom-[64px] h-1 w-5 rounded-full bg-slate-900" />
      <motion.div
        animate={{ opacity: [0.4, 1, 0.4], scale: [0.95, 1.08, 0.95] }}
        transition={{ repeat: Infinity, duration: 1.8, ease: "easeInOut" }}
        className="absolute right-4 top-4"
      >
        <Sparkles className="h-5 w-5 text-amber-400" />
      </motion.div>
    </motion.div>
  );
}

function StatCard({ title, value, subtitle, icon: Icon, accent = "from-emerald-500/20 to-lime-300/10" }) {
  return (
    <Card className="rounded-3xl border-white/50 bg-white/80 shadow-lg backdrop-blur-sm">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm text-slate-500">{title}</p>
            <h3 className="mt-2 text-3xl font-bold text-slate-900">{value}</h3>
            <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
          </div>
          <div className={`rounded-2xl bg-gradient-to-br ${accent} p-3`}>
            <Icon className="h-6 w-6 text-slate-800" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function PlantGrowthDashboard() {
  const [query, setQuery] = useState("");
  const [stageFilter, setStageFilter] = useState("All");

  const filteredData = useMemo(() => {
    return plantData.filter((row) => {
      const matchesStage = stageFilter === "All" || row.stage === stageFilter;
      const q = query.trim().toLowerCase();
      const matchesQuery =
        !q ||
        row.date.toLowerCase().includes(q) ||
        String(row.day).includes(q) ||
        row.stage.toLowerCase().includes(q);
      return matchesStage && matchesQuery;
    });
  }, [query, stageFilter]);

  const latest = filteredData[filteredData.length - 1] || plantData[plantData.length - 1];
  const firstSprout = plantData.find((d) => d.stage === "Sprout");
  const growthPercent = Math.min((latest.height / 5) * 100, 100);
  const avgTemp = Math.round(plantData.reduce((sum, d) => sum + d.temp, 0) / plantData.length);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#dcfce7,_transparent_28%),radial-gradient(circle_at_top_right,_#fde68a,_transparent_22%),linear-gradient(180deg,_#f8fafc_0%,_#eff6ff_100%)] p-6 text-slate-900">
      <div className="mx-auto max-w-7xl space-y-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="overflow-hidden rounded-[32px] border border-white/60 bg-white/70 shadow-xl backdrop-blur-md"
        >
          <div className="grid gap-6 p-8 md:grid-cols-[1.5fr_1fr]">
            <div>
              <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-emerald-100 px-3 py-1 text-sm font-medium text-emerald-700">
                <Sparkles className="h-4 w-4" /> Plant Phenotyping Dashboard
              </div>
              <h1 className="text-4xl font-black tracking-tight text-slate-900 md:text-5xl">
                Tiny sprout, big main character energy.
              </h1>
              <p className="mt-4 max-w-2xl text-base text-slate-600 md:text-lg">
                Track daily plant growth, stage transitions, and temperature trends in one fun visual dashboard.
              </p>
              <div className="mt-6 flex flex-wrap gap-3">
                <div className="relative w-full max-w-sm">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                  <Input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search day, date, or stage"
                    className="rounded-2xl border-white/60 bg-white/80 pl-9"
                  />
                </div>
                {['All', 'Seed', 'Germination', 'Sprout'].map((stage) => (
                  <Button
                    key={stage}
                    variant={stageFilter === stage ? 'default' : 'outline'}
                    className="rounded-2xl"
                    onClick={() => setStageFilter(stage)}
                  >
                    {stage}
                  </Button>
                ))}
              </div>
            </div>

            <Card className="rounded-3xl border-white/60 bg-slate-900 text-white shadow-lg">
              <CardContent className="p-6">
                <p className="text-sm uppercase tracking-[0.25em] text-emerald-300">Current vibe</p>
                <div className="mt-5 flex items-center gap-3">
                  <div className="rounded-2xl bg-emerald-500/20 p-3">
                    <Sprout className="h-7 w-7 text-emerald-300" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold">{latest.stage}</p>
                    <p className="text-sm text-slate-300">Latest recorded stage</p>
                  </div>
                </div>
                <div className="mt-4">
                  <TamagotchiPlant stage={latest.stage} />
                </div>
                <div className="mt-2 space-y-4">
                  <div>
                    <div className="mb-2 flex items-center justify-between text-sm text-slate-300">
                      <span>Growth progress</span>
                      <span>{growthPercent.toFixed(0)}%</span>
                    </div>
                    <Progress value={growthPercent} className="h-3" />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-2xl bg-white/10 p-4">
                      <p className="text-xs text-slate-300">Latest height</p>
                      <p className="mt-1 text-2xl font-bold">{latest.height} cm</p>
                    </div>
                    <div className="rounded-2xl bg-white/10 p-4">
                      <p className="text-xs text-slate-300">First sprout</p>
                      <p className="mt-1 text-2xl font-bold">Day {firstSprout?.day ?? '-'}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </motion.div>

        <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
          <StatCard title="Latest Height" value={`${latest.height} cm`} subtitle={`Recorded on ${latest.date}`} icon={Ruler} accent="from-emerald-300/40 to-green-100" />
          <StatCard title="Average Temp" value={`${avgTemp}°F`} subtitle="Across all tracked days" icon={Thermometer} accent="from-amber-300/40 to-orange-100" />
          <StatCard title="Growth Days" value={plantData.length} subtitle="Days tracked so far" icon={CalendarDays} accent="from-sky-300/40 to-cyan-100" />
          <StatCard title="Height Trend" value="Up" subtitle="Sprout is thriving" icon={TrendingUp} accent="from-violet-300/40 to-fuchsia-100" />
        </div>

        <div className="grid gap-6 xl:grid-cols-[1.4fr_1fr]">
          <Card className="rounded-3xl border-white/60 bg-white/80 shadow-lg">
            <CardHeader>
              <CardTitle>Height growth over time</CardTitle>
              <CardDescription>Watch the plant go from zero to tiny legend.</CardDescription>
            </CardHeader>
            <CardContent className="h-[340px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={filteredData}>
                  <defs>
                    <linearGradient id="heightFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="currentColor" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="currentColor" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="day" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip />
                  <Area type="monotone" dataKey="height" stroke="currentColor" fill="url(#heightFill)" strokeWidth={3} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card className="rounded-3xl border-white/60 bg-white/80 shadow-lg">
            <CardHeader>
              <CardTitle>Temperature trend</CardTitle>
              <CardDescription>How weather moved across the timeline.</CardDescription>
            </CardHeader>
            <CardContent className="h-[340px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={filteredData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="day" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip />
                  <Line type="monotone" dataKey="temp" stroke="currentColor" strokeWidth={3} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-6 xl:grid-cols-[1fr_1.2fr]">
        <Card className="rounded-3xl border-white/60 bg-white/85 shadow-lg">
          <CardHeader>
            <CardTitle>Stage buddy</CardTitle>
            <CardDescription>Your plant gets a tiny tamagotchi-style mood for each growth stage.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              {['Seed', 'Germination', 'Sprout'].map((stage) => (
                <div key={stage} className="rounded-3xl border border-slate-200 bg-gradient-to-b from-white to-slate-50 p-4 text-center shadow-sm">
                  <TamagotchiPlant stage={stage} />
                  <div className="mt-2 flex items-center justify-center gap-2">
                    <Badge className={`rounded-full border ${stageColors[stage]}`} variant="outline">
                      {stage}
                    </Badge>
                  </div>
                  <p className="mt-2 text-sm text-slate-500">
                    {stage === 'Seed' && 'sleepy bean mode'}
                    {stage === 'Germination' && 'tiny comeback arc'}
                    {stage === 'Sprout' && 'full icon behavior'}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-3xl border-white/60 bg-white/85 shadow-lg">
          <CardHeader>
            <CardTitle>Daily growth log</CardTitle>
            <CardDescription>A cute but useful overview of each recorded day.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {filteredData.map((row, index) => (
                <motion.div
                  key={row.day}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.03 }}
                  className="rounded-3xl border border-slate-200 bg-white p-4 shadow-sm"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm text-slate-500">{row.date}</p>
                      <h3 className="text-xl font-bold">Day {row.day}</h3>
                    </div>
                    <Badge className={`rounded-full border ${stageColors[row.stage]}`} variant="outline">
                      {row.stage}
                    </Badge>
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <div className="rounded-2xl bg-emerald-50 p-3">
                      <p className="text-xs text-slate-500">Height</p>
                      <p className="mt-1 text-lg font-bold">{row.height} cm</p>
                    </div>
                    <div className="rounded-2xl bg-amber-50 p-3">
                      <p className="text-xs text-slate-500">Temp</p>
                      <p className="mt-1 text-lg font-bold">{row.temp}°F</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
        </div>
      </div>
    </div>
  );
}
