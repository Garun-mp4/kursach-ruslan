import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";
import { pathToFileURL } from "node:url";

const projectDir = process.env.PROJECT_DIR;
const skillDir = process.env.SKILL_DIR;
const tempDir = process.env.TMP_DIR;
const finalPptx = process.env.FINAL_PPTX;
if (![projectDir, skillDir, tempDir, finalPptx].every((value) => path.isAbsolute(value ?? ""))) {
  throw new Error("PROJECT_DIR, SKILL_DIR, TMP_DIR and FINAL_PPTX must be absolute paths.");
}

const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href,
);
const fontFamily = resolvePresentationFont({ fontFamily: "Arial" });
const W = 1280;
const H = 720;
const C = {
  ink: "#171717",
  dark: "#272727",
  gray: "#55585C",
  mid: "#8A8A8A",
  line: "#B8B8B8",
  light: "#E5E5E5",
  pale: "#F5F5F3",
  white: "#FFFFFF",
  gold: "#C79A25",
  paleGold: "#F1E7C8",
};

const outputsDir = path.dirname(finalPptx);
const mediaDir = path.join(outputsDir, "media");
await fs.mkdir(tempDir, { recursive: true });
await fs.mkdir(outputsDir, { recursive: true });
await fs.mkdir(mediaDir, { recursive: true });
if (await exists(finalPptx)) throw new Error(`Refusing to overwrite existing deck: ${finalPptx}`);

const asset = (...parts) => path.join(projectDir, ...parts);
const videoSource = asset("06_Медиа", "M9", "single_red_center-2d771e58cc", "симуляция.mp4");
const videoTarget = path.join(mediaDir, "симуляция.mp4");
if (await exists(videoTarget)) {
  const [sourceHash, targetHash] = await Promise.all([sha256(videoSource), sha256(videoTarget)]);
  if (sourceHash !== targetHash) throw new Error("Presentation media already exists but differs from the M9 source video.");
} else {
  await fs.copyFile(videoSource, videoTarget);
}
const expectedVideoHash = "5bbed6c8d5d26fe15bfdb387c11e5eb5c1a2132a015191d2dc23abf8bb9017a3";
if ((await sha256(videoTarget)) !== expectedVideoHash) throw new Error("The MP4 does not match the verified M9 recording manifest.");

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addShape(slide, geometry, x, y, width, height, options = {}) {
  return slide.shapes.add({
    geometry,
    name: options.name,
    position: { left: x, top: y, width, height },
    fill: options.fill ?? "none",
    line: options.line ?? { style: "solid", fill: "none", width: 0 },
    ...(options.borderRadius ? { borderRadius: options.borderRadius } : {}),
  });
}

function addText(slide, text, x, y, width, height, style = {}, name) {
  const shape = addShape(slide, "textbox", x, y, width, height, { name });
  shape.text = text;
  shape.text.style = {
    typeface: fontFamily,
    fontSize: 24,
    color: C.ink,
    alignment: "left",
    verticalAlignment: "top",
    autoFit: "shrinkText",
    wrap: "square",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
    ...style,
  };
  return shape;
}

function addRule(slide, x, y, width, color = C.light, height = 1) {
  return addShape(slide, "rect", x, y, width, height, { fill: color });
}

function addImage(slide, relPath, x, y, width, height, { fit = "contain", crop, alt } = {}) {
  const bytes = fs.readFile(asset(...relPath));
  return bytes.then((imageBytes) => slide.images.add({
    blob: new Uint8Array(imageBytes),
    contentType: "image/png",
    alt: alt ?? relPath.at(-1),
    fit,
    position: { left: x, top: y, width, height },
    ...(crop ? { crop } : {}),
  }));
}

function addNotes(slide, text) {
  slide.speakerNotes.textFrame.setText(text);
}

function standardSlide(title, number, subtitle = "") {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addRule(slide, 72, 29, 1136, C.gold, 4);
  addText(slide, title, 72, 55, 1136, 64, { fontSize: 42, bold: true, color: C.ink, verticalAlignment: "middle" }, `slide-${number}-title`);
  if (subtitle) addText(slide, subtitle, 74, 121, 1120, 34, { fontSize: 21, color: C.gray, verticalAlignment: "middle" }, `slide-${number}-subtitle`);
  addRule(slide, 72, 675, 1136, C.line, 1);
  addText(slide, "КУРСОВАЯ РАБОТА · ШАТУЕВ Р.С.", 72, 684, 700, 22, { fontSize: 15, color: C.gray, verticalAlignment: "middle" }, `slide-${number}-footer`);
  addText(slide, String(number).padStart(2, "0"), 1155, 684, 53, 22, { fontSize: 15, color: C.gray, alignment: "right", verticalAlignment: "middle" }, `slide-${number}-page`);
  return slide;
}

function addBlock(slide, x, y, width, height, heading, body, name, opts = {}) {
  const border = addShape(slide, opts.geometry ?? "rect", x, y, width, height, {
    name,
    fill: opts.fill ?? C.white,
    line: { style: opts.lineStyle ?? "solid", fill: opts.stroke ?? C.ink, width: opts.lineWidth ?? 1.6 },
    ...(opts.borderRadius ? { borderRadius: opts.borderRadius } : {}),
  });
  addText(slide, heading, x + 12, y + 14, width - 24, opts.headingHeight ?? 42, {
    fontSize: opts.headingSize ?? 22,
    bold: true,
    alignment: opts.align ?? "center",
    verticalAlignment: "middle",
  }, `${name}-heading`);
  if (body) addText(slide, body, x + 12, y + (opts.bodyTop ?? 60), width - 24, height - (opts.bodyTop ?? 60) - 12, {
    fontSize: opts.bodySize ?? 18,
    color: opts.bodyColor ?? C.gray,
    alignment: opts.align ?? "center",
    verticalAlignment: "top",
  }, `${name}-body`);
  return border;
}

// 1 — cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  addRule(slide, 90, 48, 1100, C.gold, 4);
  addText(slide, "ФЕДЕРАЛЬНОЕ АГЕНТСТВО РОССИЙСКОЙ ФЕДЕРАЦИИ ПО РЫБОЛОВСТВУ", 100, 82, 1080, 30, {
    fontSize: 21, bold: true, alignment: "center", verticalAlignment: "middle",
  }, "cover-agency");
  addText(slide, "АСТРАХАНСКИЙ ГОСУДАРСТВЕННЫЙ ТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ", 100, 122, 1080, 32, {
    fontSize: 23, bold: true, alignment: "center", verticalAlignment: "middle",
  }, "cover-university");
  addText(slide, "Институт информационных технологий и коммуникаций\nКафедра автоматизированных систем обработки информации и управления", 150, 165, 980, 56, {
    fontSize: 19, alignment: "center", verticalAlignment: "middle", color: C.gray,
  }, "cover-department");
  addText(slide, "КУРСОВАЯ РАБОТА", 140, 257, 1000, 44, {
    fontSize: 32, bold: true, alignment: "center", verticalAlignment: "middle",
  }, "cover-type");
  addText(slide, "Моделирование и симуляция робота-манипулятора,\nвыполняющего задачу сортировки объектов по цвету", 120, 318, 1040, 142, {
    fontSize: 43, bold: true, alignment: "center", verticalAlignment: "middle",
  }, "cover-topic");
  addRule(slide, 395, 481, 490, C.gold, 3);
  addText(slide, "по дисциплине «Моделирование роботов»", 150, 496, 980, 38, {
    fontSize: 23, alignment: "center", verticalAlignment: "middle", color: C.gray,
  }, "cover-subject");
  addText(slide, "Выполнил: Шатуев Руслан Саматович\nГруппа: ДИНРб-31", 155, 563, 500, 66, {
    fontSize: 21, verticalAlignment: "middle",
  }, "cover-student");
  addText(slide, "Руководитель: Кузнецова В.Ю.", 660, 563, 465, 66, {
    fontSize: 21, alignment: "right", verticalAlignment: "middle",
  }, "cover-advisor");
  addText(slide, "Астрахань — 2026", 390, 645, 500, 28, {
    fontSize: 20, bold: true, alignment: "center", verticalAlignment: "middle",
  }, "cover-city-year");
  addNotes(slide, "Время: 0:15. Представить тему и автора. В работе рассматривается программная модель манипулятора для сортировки объектов по цвету.\nИсточник реквизитов: 01_Управление/реквизиты.yaml. Год оформления записки — 2026; дата защиты отдельно не подтверждена.");
}

// 2 — objective and scope
{
  const slide = standardSlide("Цель — разработать и проверить виртуальную ячейку сортировки", 2);
  addText(slide, "Манипулятор должен определить цвет и положение объекта по RGB-кадру, захватить его и перенести в ячейку соответствующего класса.", 82, 164, 1110, 74, {
    fontSize: 27, bold: true, verticalAlignment: "middle",
  }, "goal-statement");
  addRule(slide, 82, 252, 1110, C.line, 1);
  const tasks = [
    ["01", "Задать геометрию", "SCARA, рабочая ячейка, ограничения суставов"],
    ["02", "Рассчитать движение", "FK/IK, траектория и контроль столкновений"],
    ["03", "Распознать объект", "RGB-кадр, цвет, XY и ориентация"],
    ["04", "Выполнить сортировку", "Контактный захват, FSM, проверка укладки"],
  ];
  const xs = [82, 645];
  const ys = [286, 423];
  tasks.forEach(([num, heading, body], i) => {
    const x = xs[i % 2];
    const y = ys[Math.floor(i / 2)];
    addText(slide, num, x, y, 62, 42, { fontSize: 32, bold: true, color: C.gold, verticalAlignment: "middle" }, `goal-${num}-number`);
    addText(slide, heading, x + 78, y, 450, 38, { fontSize: 25, bold: true, verticalAlignment: "middle" }, `goal-${num}-title`);
    addText(slide, body, x + 78, y + 44, 455, 56, { fontSize: 21, color: C.gray }, `goal-${num}-body`);
  });
  addRule(slide, 82, 558, 1110, C.light, 1);
  addText(slide, "Физический прототип не изготовлялся. Проверка и результаты относятся только к программной симуляции.", 82, 574, 1110, 52, {
    fontSize: 22, bold: true, color: C.gray, verticalAlignment: "middle",
  }, "goal-scope");
  addNotes(slide, "Время: 0:35. Сформулировать цель как работу программной ячейки, а не как проектирование уже изготовленного устройства. Перечислить четыре задачи: геометрическая модель, кинематика и траектории, распознавание по изображению, выполнение и проверка цикла. Отдельно обозначить границу: физическая сборка и натурные испытания не выполнялись.\nИсточник: 01_Управление/Границы_проекта.md; 08_Записка/source/Пояснительная_записка.md, введение и разделы 1–2.");
}

// 3 — architecture
{
  const slide = standardSlide("Архитектура разделяет управление и независимую оценку", 3);
  addText(slide, "УПРАВЛЯЮЩИЙ КОНТУР · только публичные наблюдения и состояние суставов", 82, 165, 1115, 30, {
    fontSize: 18, bold: true, color: C.gray, verticalAlignment: "middle",
  }, "architecture-loop-label");
  const blocks = [
    ["RGB-камеры", "overhead_rgb\nplacement_rgb"],
    ["Perception", "цвет · XY · yaw\nconfidence / UNKNOWN"],
    ["Кинематика и путь", "FK / IK\nпроверка столкновений"],
    ["Контроллер / FSM", "выбор · захват\nперенос · отказ"],
    ["MuJoCo", "динамика\nконтакты и суставы"],
  ];
  const x0 = 82;
  const y = 224;
  const w = 195;
  const h = 154;
  const gap = 42;
  const shapes = blocks.map(([head, body], i) => addBlock(slide, x0 + i * (w + gap), y, w, h, head, body, `architecture-block-${i + 1}`, {
    headingSize: 22, bodySize: 18, bodyTop: 70,
  }));
  for (let i = 0; i < shapes.length - 1; i++) {
    slide.shapes.connect(shapes[i], shapes[i + 1], {
      kind: "straight", fromSide: "right", toSide: "left",
      line: { style: "solid", fill: C.ink, width: 2 },
      tail: { type: "arrow", width: "sm", length: "sm" },
    });
  }
  addText(slide, "Команды суставам", 928, 394, 165, 24, { fontSize: 16, color: C.gray, alignment: "center", verticalAlignment: "middle" }, "architecture-command-label");
  addText(slide, "Независимый evaluator", 82, 468, 245, 34, { fontSize: 23, bold: true, verticalAlignment: "middle" }, "evaluator-title");
  const evaluator = addShape(slide, "rect", 82, 511, 1115, 88, {
    name: "evaluator-boundary",
    fill: C.pale,
    line: { style: "dashed", fill: C.ink, width: 1.7 },
  });
  addText(slide, "После прогона сверяет истинный цвет, зону, слот, контакты и ограничения. Ground truth не поступает в контроллер.", 105, 530, 1065, 50, {
    fontSize: 22, color: C.dark, verticalAlignment: "middle",
  }, "evaluator-description");
  addNotes(slide, "Время: 0:40. Проследить поток слева направо: камеры отдают изображение, perception оценивает объект, IK и планировщик проверяют движение, FSM выдает команды MuJoCo. Нижняя пунктирная область изолирована: истинные значения используются только после действий независимым evaluator, поэтому контроллер не получает правильный ответ из движка.\nИсточник: 02_Спецификация/Архитектура.md, раздел 5 и 7; 08_Записка/figures/01_Архитектура_M1.png.");
}

// 4 — workcell layout
{
  const slide = standardSlide("Рабочая ячейка объединяет подачу и три зоны сортировки", 4);
  await addImage(slide, ["08_Записка", "figures", "03_Компоновка_ячейки_M7.png"], 78, 165, 548, 486, {
    fit: "contain",
    crop: { left: 0, top: 0, right: 0.40, bottom: 0 },
    alt: "Вид сверху на стол, SCARA, подачу и лотки; фрагмент размерной компоновки M7",
  });
  addText(slide, "СТАТИЧНАЯ СЦЕНА", 680, 176, 490, 30, { fontSize: 18, bold: true, color: C.gold, verticalAlignment: "middle" }, "cell-kicker");
  const cellFacts = [
    ["Стол", "1000 × 900 мм"],
    ["Вход", "6 объектов · 2 на класс"],
    ["Сортировка", "RED · GREEN · BLUE"],
    ["Лотки", "3 зоны · по 3 слота"],
    ["Объект", "28 × 28 × 16 мм"],
    ["Наблюдение", "две фиксированные RGB-камеры"],
  ];
  cellFacts.forEach(([label, value], i) => {
    const yy = 227 + i * 62;
    addText(slide, label.toUpperCase(), 680, yy, 165, 26, { fontSize: 16, bold: true, color: C.gray, verticalAlignment: "middle" }, `cell-label-${i}`);
    addText(slide, value, 680, yy + 24, 495, 34, { fontSize: 22, bold: i < 2, verticalAlignment: "middle" }, `cell-value-${i}`);
    if (i < cellFacts.length - 1) addRule(slide, 680, yy + 59, 495, C.light, 1);
  });
  addNotes(slide, "Время: 0:40. Показать расположение подачи и лотков относительно основания манипулятора. Это фиксированная рабочая ячейка: объекты лежат одним слоем, а лотки имеют выделенные места. Размеры и число объектов взяты из M2/M7, не масштабирвать рисунок как точный чертеж.\nИсточники: 02_Спецификация/параметры_системы.yaml (M7-v1.8); 08_Записка/figures/03_Компоновка_ячейки_M7.png; 08_Записка/source/Пояснительная_записка.md, раздел 3.");
}

// 5 — physical/visual model
{
  const slide = standardSlide("В модели задан SCARA с контактным двухпальцевым захватом", 5);
  await addImage(slide, ["08_Записка", "figures", "02_SCARA_изометрия_M7.png"], 76, 161, 700, 472, {
    fit: "contain",
    alt: "Изометрический рендер программной модели SCARA и сортировочной ячейки",
  });
  addText(slide, "4", 840, 172, 112, 72, { fontSize: 62, bold: true, verticalAlignment: "middle" }, "model-four-dof-number");
  addText(slide, "степени свободы руки", 953, 185, 240, 48, { fontSize: 23, bold: true, verticalAlignment: "middle" }, "model-four-dof-label");
  addRule(slide, 840, 260, 350, C.line, 1);
  const modelMetrics = [
    ["R–R–P–R", "две оси XY · Z-ползун · yaw"],
    ["230 / 200 мм", "длины звеньев L1 / L2"],
    ["+ 1 координата", "симметричный привод захвата"],
    ["28 × 28 × 16 мм", "габарит сортируемой детали"],
  ];
  modelMetrics.forEach(([value, label], i) => {
    const yy = 282 + i * 78;
    addText(slide, value, 840, yy, 350, 34, { fontSize: 24, bold: true, verticalAlignment: "middle" }, `model-value-${i}`);
    addText(slide, label, 840, yy + 36, 350, 30, { fontSize: 19, color: C.gray, verticalAlignment: "middle" }, `model-label-${i}`);
  });
  addNotes(slide, "Время: 0:35. Изометрия показывает программную геометрию, а не фотографию изготовленного устройства. Рука имеет четыре независимые координаты; параллельные пальцы связаны механически и управляются одной дополнительной координатой. Длины звеньев и габариты детали даны в SSOT.\nИсточники: 02_Спецификация/Архитектура.md; 02_Спецификация/параметры_системы.yaml; 08_Записка/figures/02_SCARA_изометрия_M7.png.");
}

// 6 — kinematics
{
  const slide = standardSlide("Кинематика связывает суставы с положением и yaw инструмента", 6);
  addText(slide, "q = [ q₁, q₂, q₃, q₄ ]", 82, 167, 560, 45, { fontSize: 29, bold: true, verticalAlignment: "middle" }, "kinematics-q");
  addText(slide, "R · R · P · R", 708, 167, 440, 45, { fontSize: 29, bold: true, color: C.gray, alignment: "right", verticalAlignment: "middle" }, "kinematics-types");
  addRule(slide, 82, 228, 1080, C.line, 1);
  addText(slide,
    "x = x_b + L₁ cos q₁ + L₂ cos(q₁ + q₂)\ny = y_b + L₁ sin q₁ + L₂ sin(q₁ + q₂)\nz = z_b + 0,103 − q₃\nψ = wrap(q₁ + q₂ + q₄)",
    85, 257, 700, 220,
    { fontSize: 25, color: C.ink, verticalAlignment: "middle", lineSpacing: 1.18 },
    "kinematics-fk-formulas");
  addRule(slide, 815, 256, 2, C.gold, 235);
  addText(slide, "ОБРАТНАЯ КИНЕМАТИКА", 852, 264, 320, 36, { fontSize: 18, bold: true, color: C.gray, verticalAlignment: "middle" }, "kinematics-ik-label");
  const ikSteps = [
    "2 ветви решения локтя",
    "отсев по лимитам суставов",
    "контроль остатка независимым FK",
  ];
  ikSteps.forEach((item, i) => {
    addText(slide, `${i + 1}.`, 852, 321 + i * 58, 32, 34, { fontSize: 22, bold: true, color: C.gold, verticalAlignment: "middle" }, `ik-num-${i}`);
    addText(slide, item, 892, 321 + i * 58, 285, 46, { fontSize: 20, verticalAlignment: "middle" }, `ik-step-${i}`);
  });
  addText(slide, "L₁ = 0,230 м · L₂ = 0,200 м", 82, 533, 1080, 38, { fontSize: 22, bold: true, color: C.gray, alignment: "center", verticalAlignment: "middle" }, "kinematics-links");
  addNotes(slide, "Время: 0:45. Объяснить структуру R–R–P–R: первые два поворотных сочленения задают XY, вертикальный ползун — Z, запястье — yaw. Формулы — прямая кинематика TCP в координатах BASE с учетом базового смещения. При обратной задаче аналитически рассматриваются обе ветви локтя, кандидаты фильтруются по физическим пределам, затем решение проверяется повторным FK.\nИсточник: 02_Спецификация/Кинематика.md, разделы 2–5; 04_Программа/kinematics/scara.py; длины из 02_Спецификация/параметры_системы.yaml.");
}

// 7 — perception
{
  const slide = standardSlide("Цвет и положение оцениваются по публичному RGB-кадру", 7, "Классификатор использует изображение; истинная метка остается только у evaluator");
  await addImage(slide, ["06_Медиа", "M9", "single_red_center-2d771e58cc", "кадры", "detection.png"], 72, 174, 716, 403, {
    fit: "contain",
    alt: "Кадр виртуальной камеры и визуализация состояния контроллера на стадии perception",
  });
  addText(slide, "V03 · ПЕРВЫЕ RGB-КАДРЫ", 833, 187, 353, 30, { fontSize: 18, bold: true, color: C.gray, verticalAlignment: "middle" }, "perception-label");
  addText(slide, "90 / 90", 833, 226, 353, 72, { fontSize: 57, bold: true, verticalAlignment: "middle" }, "perception-classes");
  addText(slide, "правильных цветовых классификаций", 833, 298, 353, 48, { fontSize: 21, color: C.gray, verticalAlignment: "middle" }, "perception-class-label");
  addRule(slide, 833, 364, 350, C.line, 1);
  addText(slide, "0,415 мм", 833, 385, 353, 48, { fontSize: 32, bold: true, verticalAlignment: "middle" }, "perception-xy");
  addText(slide, "средняя ошибка XY на той же выборке", 833, 434, 353, 45, { fontSize: 19, color: C.gray, verticalAlignment: "middle" }, "perception-xy-label");
  addText(slide, "Показатель относится к 90 объектам V03 при заданной палитре и условиях; это не оценка реальной камеры.", 833, 510, 353, 78, { fontSize: 18, color: C.gray }, "perception-caveat");
  addNotes(slide, "Время: 0:40. Показать входной сенсорный кадр и отдельно пояснить границу данных: контроллер классифицирует RGB-наблюдение, evaluator после запуска сравнивает его с эталоном. На первых публичных кадрах 30 RED, 30 GREEN и 30 BLUE объектов классифицированы правильно. Средняя XY-ошибка 0,415 мм, максимум 1,316 мм; значение относится к ограниченным условиям кампании.\nИсточники: 07_Испытания/campaign_results.json, baseline_first_view_color; 07_Испытания/Анализ.md; 06_Медиа/M9/single_red_center-2d771e58cc/кадры/detection.png.");
}

// 8 — motion planning
{
  const slide = standardSlide("Планировщик проверяет весь путь до выдачи команд", 8);
  await addImage(slide, ["08_Записка", "figures", "10_Траектория_M9.png"], 70, 165, 620, 473, {
    fit: "contain",
    alt: "Измеренная траектория TCP в представительном виртуальном прогоне M9",
  });
  addText(slide, "ПОЛНЫЙ ПРЕДВАРИТЕЛЬНЫЙ КОНТРОЛЬ", 742, 174, 445, 31, { fontSize: 18, bold: true, color: C.gold, verticalAlignment: "middle" }, "planner-label");
  const planSteps = [
    ["Цель", "позиция и ориентация TCP"],
    ["IK", "допустимые ветви и лимиты"],
    ["Траектория", "плавные суставные профили"],
    ["Коллизии", "звенья, пальцы и переносимый объект"],
  ];
  planSteps.forEach(([head, body], i) => {
    const yy = 225 + i * 77;
    addText(slide, String(i + 1).padStart(2, "0"), 742, yy, 43, 39, { fontSize: 21, bold: true, color: C.gold, verticalAlignment: "middle" }, `planner-${i}-num`);
    addText(slide, head, 795, yy, 392, 34, { fontSize: 22, bold: true, verticalAlignment: "middle" }, `planner-${i}-head`);
    addText(slide, body, 795, yy + 34, 392, 32, { fontSize: 18, color: C.gray, verticalAlignment: "middle" }, `planner-${i}-body`);
  });
  addRule(slide, 742, 546, 445, C.line, 1);
  addText(slide, "Если безопасный путь не найден, движение по этому плану не начинается.", 742, 559, 445, 64, { fontSize: 20, bold: true, verticalAlignment: "middle" }, "planner-refusal");
  addNotes(slide, "Время: 0:40. Траектория на графике — фактическая TCP-телеметрия одного зарегистрированного успешного прогона, а не средняя по кампании. Планировщик проверяет суставные и скоростные ограничения и геометрию всей руки, захвата и груза. Проверка дискретная и не дает непрерывной формальной гарантии отсутствия контакта между отсчетами. При неприемлемом пути контроллер должен отказаться от движения к цели.\nИсточники: 02_Спецификация/Траектории_и_коллизии.md; 07_Испытания/Анализ.md, разделы 2 и 3; 08_Записка/figures/10_Траектория_M9.png.");
}

// 9 — sorter state machine
{
  const slide = standardSlide("FSM связывает наблюдение, захват и проверку укладки", 9);
  addText(slide, "ОСНОВНОЙ ЦИКЛ · UML state abstraction", 82, 165, 1080, 28, { fontSize: 18, bold: true, color: C.gray, verticalAlignment: "middle" }, "fsm-kicker");
  const stateY = 233;
  const stateW = 158;
  const stateH = 105;
  const stateGap = 32;
  const stateStart = 93;
  const stateInfo = [
    ["OBSERVE", "свежий кадр"],
    ["SELECT / PLAN", "цель и слот"],
    ["APPROACH /\nGRASP", "контакт"],
    ["VERIFY_HOLD", "проверка удержания"],
    ["TRANSFER /\nPLACE", "перенос и отпускание"],
    ["VERIFY_\nPLACE", "свежая RGB-проверка"],
  ];
  const stateShapes = stateInfo.map(([head, body], i) => addBlock(slide, stateStart + i * (stateW + stateGap), stateY, stateW, stateH, head, body, `fsm-state-${i}`, {
    geometry: "roundRect", borderRadius: 12, headingSize: 18, headingHeight: 46, bodyTop: 58, bodySize: 16, lineWidth: 1.8,
  }));
  const initial = addShape(slide, "ellipse", 71, stateY + 44, 16, 16, { name: "fsm-initial", fill: C.ink, line: { style: "solid", fill: C.ink, width: 1 } });
  slide.shapes.connect(initial, stateShapes[0], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.ink, width: 2 }, tail: { type: "arrow", width: "sm", length: "sm" } });
  for (let i = 0; i < stateShapes.length - 1; i++) {
    slide.shapes.connect(stateShapes[i], stateShapes[i + 1], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.ink, width: 2 }, tail: { type: "arrow", width: "sm", length: "sm" } });
  }
  slide.shapes.connect(stateShapes[5], stateShapes[0], { kind: "elbow", fromSide: "bottom", toSide: "bottom", line: { style: "solid", fill: C.ink, width: 1.8 }, tail: { type: "arrow", width: "sm", length: "sm" } });
  addText(slide, "новый кадр", 110, 374, 145, 25, { fontSize: 16, color: C.gray, alignment: "center", verticalAlignment: "middle" }, "fsm-loop-label");
  addRule(slide, 82, 433, 1115, C.line, 1);
  addText(slide, "БЕЗОПАСНЫЕ ВЕТВИ", 82, 457, 255, 30, { fontSize: 18, bold: true, color: C.gray, verticalAlignment: "middle" }, "fsm-errors-label");
  const failCases = [
    ["UNKNOWN / недоступен", "пропуск с причиной"],
    ["камера / удержание", "ограниченный recovery или SAFE_STOP"],
    ["укладка не подтверждена", "слот в карантин · без ложного успеха"],
  ];
  failCases.forEach(([head, body], i) => {
    const x = 82 + i * 373;
    addText(slide, head, x, 506, 345, 35, { fontSize: 21, bold: true, verticalAlignment: "middle" }, `fsm-failure-${i}-head`);
    addText(slide, body, x, 544, 345, 48, { fontSize: 18, color: C.gray, verticalAlignment: "middle" }, `fsm-failure-${i}-body`);
    if (i < 2) addRule(slide, x + 355, 502, 1, C.light, 87);
  });
  addNotes(slide, "Время: 0:45. Показать основной цикл: свежее наблюдение, выбор цели/слота и preflight, подход, захват, проверка удержания, перенос и проверка укладки свежей RGB-камерой. Схема намеренно верхнеуровневая; полная диаграмма состояний и guards остаются в записке. UNKNOWN, недоступный объект и занятый слот не превращаются в ложную сортировку; сбой камеры или удержания вызывает ограниченную реакцию или SAFE_STOP.\nИсточники: 02_Спецификация/FSM.md; 08_Записка/figures/07_FSM_M6.png; 08_Записка/figures/14_Исполнение_и_проверка_укладки_M6.png.");
}

// 10 — close-up cycle evidence
{
  const slide = standardSlide("Контактный цикл проверяет захват, перенос и отпускание", 10);
  const frames = [
    ["grasp.png", "Захват"],
    ["transfer.png", "Перенос"],
    ["release.png", "Отпускание"],
  ];
  for (let i = 0; i < frames.length; i++) {
    const x = 82 + i * 378;
    await addImage(slide, ["06_Медиа", "M9", "single_red_center-2d771e58cc", "кадры", frames[i][0]], x, 179, 344, 194, {
      fit: "contain",
      alt: `Кадр виртуального прогона M9: ${frames[i][1].toLowerCase()}`,
    });
    addText(slide, frames[i][1], x, 382, 344, 34, { fontSize: 22, bold: true, alignment: "center", verticalAlignment: "middle" }, `cycle-frame-label-${i}`);
  }
  addRule(slide, 82, 440, 1110, C.line, 1);
  const checks = [
    "двусторонний контакт пальцев",
    "сигнал удержания на подъеме",
    "свежий кадр после отхода",
  ];
  checks.forEach((label, i) => {
    const x = 82 + i * 378;
    addText(slide, `0${i + 1}`, x, 477, 48, 42, { fontSize: 27, bold: true, color: C.gold, verticalAlignment: "middle" }, `cycle-check-num-${i}`);
    addText(slide, label, x + 58, 475, 284, 56, { fontSize: 20, verticalAlignment: "middle" }, `cycle-check-${i}`);
  });
  addText(slide, "Это кадры одной программной записи; реального захвата физической рукой не было.", 82, 579, 1110, 38, { fontSize: 19, color: C.gray, alignment: "center", verticalAlignment: "middle" }, "cycle-disclaimer");
  addNotes(slide, "Время: 0:35. Проследить три реальные стадии одной записанной виртуальной симуляции. Контактное удержание контролируется по сигналам пальцев и проверяется при подъеме; отпускание подтверждается отдельным RGB-кадром после безопасного отхода. Кадры иллюстрируют цифровой прогон, не физический прототип.\nИсточники: 06_Медиа/M9/single_red_center-2d771e58cc/Запись_и_связанные_логи.json; 07_Испытания/runs/single_red_center-2d771e58cc/evaluator/report.json; 08_Записка/figures/14_Исполнение_и_проверка_укладки_M6.png.");
}

// 11 — separate MP4
{
  const slide = standardSlide("Отдельная запись показывает один завершенный виртуальный цикл", 11);
  await addImage(slide, ["08_Записка", "figures", "12_Кадр_успешного_виртуального_цикла.png"], 72, 166, 715, 402, {
    fit: "contain",
    alt: "Кадр завершенного виртуального цикла, run single_red_center-2d771e58cc",
  });
  addText(slide, "ЗАПИСАННЫЙ ПРОГОН", 831, 174, 360, 30, { fontSize: 18, bold: true, color: C.gold, verticalAlignment: "middle" }, "video-kicker");
  addText(slide, "single_red_center-2d771e58cc", 831, 217, 360, 52, { fontSize: 20, bold: true, verticalAlignment: "middle" }, "video-run-id");
  addText(slide, "1 объект · RED · FSM DONE", 831, 280, 360, 40, { fontSize: 21, verticalAlignment: "middle" }, "video-result");
  addText(slide, "33,04 с видео\nоколо 5× к модельному времени", 831, 336, 360, 70, { fontSize: 22, verticalAlignment: "middle" }, "video-duration");
  addRule(slide, 831, 426, 360, C.line, 1);
  const videoLink = addText(slide, "Открыть MP4: media/симуляция.mp4", 831, 449, 360, 52, {
    fontSize: 20, color: C.ink, verticalAlignment: "middle",
  }, "video-relative-link");
  const linkRange = videoLink.text.get("Открыть MP4: media/симуляция.mp4");
  linkRange.link = { uri: "media/симуляция.mp4", isExternal: true };
  linkRange.underline = "sng";
  addText(slide, "MP4 хранится рядом с презентацией. Если ссылки отключены — откройте файл из папки media.", 831, 511, 360, 72, {
    fontSize: 18, color: C.gray,
  }, "video-portability");
  addText(slide, "Эта запись демонстрирует один успешный run и не заменяет результаты кампании M9.", 82, 599, 1110, 38, {
    fontSize: 19, bold: true, color: C.gray, alignment: "center", verticalAlignment: "middle",
  }, "video-scope");
  addNotes(slide, "Время: 0:30. Это один отдельный успешный прогон single_red_center: один красный объект, FSM завершился DONE. Видео связано с записьным parity-run и воспроизводимой конфигурацией, длительность 33,04 с при эффективном ускорении примерно 5×. Файл не встраивается в PPTX: рядом сохраняется MP4 и относительная ссылка, чтобы не раздувать презентацию; в приложении защиты ссылка еще не проверена. Не представлять этот run как среднее или итог кампании.\nИсточники: 06_Медиа/M9/single_red_center-2d771e58cc/Запись_и_связанные_логи.json; 06_Медиа/M9/single_red_center-2d771e58cc/Сверка_с_записью.json; MP4 в media/симуляция.mp4.");
}

// 12 — test protocol
{
  const slide = standardSlide("Кампания M9 включала 98 запусков и заранее заданные критерии", 12);
  const metrics = [
    ["98 / 98", "зарегистрированных запусков\nпроанализированы"],
    ["30", "независимых партий\nв основной серии V03"],
    ["27 / 30", "порог приемки V03\nне менее 90% партий"],
  ];
  const xPositions = [88, 474, 860];
  metrics.forEach(([value, label], i) => {
    const x = xPositions[i];
    addText(slide, value, x, 204, 320, 86, { fontSize: 55, bold: true, verticalAlignment: "middle" }, `test-metric-${i}`);
    addText(slide, label, x, 299, 320, 68, { fontSize: 22, color: C.gray, verticalAlignment: "middle" }, `test-label-${i}`);
    if (i < metrics.length - 1) addRule(slide, x + 347, 205, 1, C.light, 184);
  });
  addRule(slide, 88, 404, 1090, C.line, 1);
  addText(slide, "Сценарии: фиксированные цвета · смешанные партии · крайние позиции · шум RGB · освещение · отказ камеры и захвата · заблокированный путь · повторяемость", 88, 432, 1090, 88, {
    fontSize: 22, verticalAlignment: "middle",
  }, "test-scenarios");
  addText(slide, "84/98 запусков соответствуют ожидаемому исходу, включая корректные безопасные отказы; это не доля успешной сортировки.", 88, 548, 1090, 56, {
    fontSize: 20, bold: true, color: C.gray, verticalAlignment: "middle",
  }, "test-outcomes-note");
  addNotes(slide, "Время: 0:40. Описать кампанию по зарегистрированному протоколу M9: 98 запусков проанализированы, основной V03 — 30 независимых трехобъектных партий с заранее установленным порогом 27 полных партий. 84/98 — соответствие ожидаемому исходу для разных функциональных и стресс-тестов; сюда входят корректные безопасные отказы, поэтому показатель нельзя называть успешной сортировкой.\nИсточники: 07_Испытания/campaign.yaml; 07_Испытания/campaign_results.json; 01_Управление/приемка/M09.md.");
}

// 13 — primary result
{
  const slide = standardSlide("V03: завершено 21 из 30 партий при цели 27", 13, "Полная партия — все три объекта подтверждены evaluator в верных слотах, FSM заканчивается DONE");
  const x0 = 125;
  const x1 = 1130;
  const barY = 325;
  const barH = 62;
  const scaleX = (percent) => x0 + (x1 - x0) * percent / 100;
  addText(slide, "0%", x0 - 13, 418, 62, 27, { fontSize: 17, color: C.gray, verticalAlignment: "middle" }, "result-axis-0");
  addText(slide, "50%", scaleX(50) - 24, 418, 70, 27, { fontSize: 17, color: C.gray, alignment: "center", verticalAlignment: "middle" }, "result-axis-50");
  addText(slide, "100%", x1 - 42, 418, 67, 27, { fontSize: 17, color: C.gray, alignment: "right", verticalAlignment: "middle" }, "result-axis-100");
  addShape(slide, "rect", x0, barY, x1 - x0, barH, { name: "result-track", fill: C.pale, line: { style: "solid", fill: C.line, width: 1 } });
  addShape(slide, "rect", x0, barY, scaleX(70) - x0, barH, { name: "result-actual", fill: "#9A9A9A", line: { style: "solid", fill: C.ink, width: 1 } });
  addShape(slide, "rect", scaleX(90) - 2, barY - 31, 4, barH + 64, { name: "result-target-marker", fill: C.gold });
  addText(slide, "21 / 30 = 70,0%", x0 + 12, barY + 9, 380, 42, { fontSize: 27, bold: true, verticalAlignment: "middle" }, "result-actual-label");
  addText(slide, "цель 90%", scaleX(90) - 80, 267, 160, 32, { fontSize: 19, bold: true, color: C.ink, alignment: "center", verticalAlignment: "middle" }, "result-target-label");
  // Wilson 95% interval for the independent three-object batch rate.
  const ciY = 399;
  addShape(slide, "rect", scaleX(52.1), ciY, scaleX(83.3) - scaleX(52.1), 3, { name: "result-wilson-line", fill: C.ink });
  addShape(slide, "rect", scaleX(52.1) - 2, ciY - 7, 4, 17, { name: "result-wilson-low", fill: C.ink });
  addShape(slide, "rect", scaleX(83.3) - 2, ciY - 7, 4, 17, { name: "result-wilson-high", fill: C.ink });
  addText(slide, "95% интервал Уилсона: 52,1–83,3%", 310, 463, 630, 34, { fontSize: 19, color: C.gray, alignment: "center", verticalAlignment: "middle" }, "result-interval");
  addRule(slide, 88, 516, 1090, C.line, 1);
  addText(slide, "73 / 90 объектов размещены верно", 88, 539, 400, 44, { fontSize: 23, bold: true, verticalAlignment: "middle" }, "result-object-success");
  addText(slide, "0 неверных зон · 0 запрещенных контактов · 0 нарушений суставных/силовых лимитов в V03", 506, 539, 672, 54, { fontSize: 19, color: C.gray, verticalAlignment: "middle" }, "result-safety-metrics");
  addNotes(slide, "Время: 0:55. Единица анализа — полная трехобъектная партия, а не один объект. Получено 21/30 или 70%, при заранее установленном пороге 27/30 (90%); Wilson 95% интервал 52,1–83,3%. Поэтому основная функциональная цель не достигнута. Отдельно: 73/90 объектов попали в верную зону; в этой именно серии evaluator не зарегистрировал wrong-bin, запрещенных контактов и превышений ограничений. Ноль нарушений безопасности не превращает безопасно отклоненную цель в сортировку.\nИсточник: 07_Испытания/campaign_results.json, primary_baseline и series[V03]; 07_Испытания/Анализ.md, раздел 2; 08_Записка/tables/Сводка_M9_для_записки.csv.");
}

// 14 — negative cases
{
  const slide = standardSlide("Негативные прогоны показали ограничения захвата и проверки укладки", 14);
  const cols = [
    ["V03 · ПЛАНИРОВАНИЕ", "17 / 90", "объектов не взяты\nпланировщик отказал: COLLISION_GRASP", "Без движения по отклоненной цели; поэтому неполные партии."],
    ["V02 · КОНТАКТ", "3 / 6", "подтвержденных укладок\n1 запрещенный контакт при размещении", "FSM завершился SAFE_STOP / PUBLIC_HOLD_SIGNAL_LOST."],
    ["V07 · ОСВЕЩЕНИЕ", "2 прогона", "при 0,70 и 1,30 освещенности\nпроверка укладки: timeout", "Evaluator видел правильную укладку; контроллер ее не подтвердил."],
  ];
  cols.forEach(([head, value, detail, takeaway], i) => {
    const x = 82 + i * 375;
    addText(slide, head, x, 182, 345, 32, { fontSize: 17, bold: true, color: C.gold, verticalAlignment: "middle" }, `negative-${i}-heading`);
    addText(slide, value, x, 232, 345, 66, { fontSize: 42, bold: true, verticalAlignment: "middle" }, `negative-${i}-value`);
    addText(slide, detail, x, 316, 345, 110, { fontSize: 20, verticalAlignment: "top" }, `negative-${i}-detail`);
    addRule(slide, x, 440, 345, C.line, 1);
    addText(slide, takeaway, x, 462, 345, 110, { fontSize: 19, color: C.gray, verticalAlignment: "top" }, `negative-${i}-takeaway`);
    if (i < cols.length - 1) addRule(slide, x + 360, 183, 1, C.light, 394);
  });
  addText(slide, "Отказ планировщика помог избежать опасного движения, но снизил полноту сортировки.", 82, 600, 1110, 36, {
    fontSize: 20, bold: true, alignment: "center", verticalAlignment: "middle",
  }, "negative-conclusion");
  addNotes(slide, "Время: 0:55. Разделить три вида ограничения. В V03 17 объектов были безопасно отклонены планировщиком из-за COLLISION_GRASP, команды для этих целей не выдавались. В V02 шесть объектов обнаружены, три подтверждены evaluator, но случился один контакт пальца с ранее размещенной деталью; автомат перешел в SAFE_STOP. В двух световых стресс-прогонах evaluator подтвердил правильную укладку, но публичная камера и контроллер не подтвердили ее и выдали PLACE_VERIFICATION_TIMEOUT. Это ограничение проверки, а не ошибка цветового класса.\nИсточники: 07_Испытания/Анализ.md, разделы 2–4; 07_Испытания/campaign_results.json; `07_Испытания/runs/m9_six_object_grid-0ce3eda3ca/evaluator/report.json`.");
}

// 15 — conclusion
{
  const slide = standardSlide("Итог: виртуальная сортировка реализована, критерий полноты не достигнут", 15);
  const conclusions = [
    ["Модель", "SCARA R–R–P–R · две RGB-камеры · контактный захват"],
    ["Алгоритм", "аналитическая FK/IK · perception · collision-aware планирование · FSM"],
    ["Проверка", "98 виртуальных запусков; V14 подтвердил повторяемость одного seed"],
  ];
  conclusions.forEach(([head, body], i) => {
    const y = 190 + i * 97;
    addText(slide, head.toUpperCase(), 88, y, 178, 33, { fontSize: 17, bold: true, color: C.gold, verticalAlignment: "middle" }, `conclusion-${i}-head`);
    addText(slide, body, 280, y, 890, 48, { fontSize: 24, bold: i === 0, verticalAlignment: "middle" }, `conclusion-${i}-body`);
    if (i < conclusions.length - 1) addRule(slide, 88, y + 69, 1080, C.light, 1);
  });
  addRule(slide, 88, 486, 1080, C.line, 1);
  addText(slide, "V03: 21/30 полных партий при цели 27/30. Следующая инженерная задача — повысить доступность безопасного захвата и надежность проверки укладки.", 88, 508, 1080, 74, {
    fontSize: 24, bold: true, verticalAlignment: "middle",
  }, "conclusion-outcome");
  addText(slide, "Выводы относятся к модели MuJoCo; физический прототип и натурные измерения отсутствуют.", 88, 599, 1080, 35, {
    fontSize: 20, color: C.gray, alignment: "center", verticalAlignment: "middle",
  }, "conclusion-scope");
  addNotes(slide, "Время: 0:35. Подвести итог: создана программная ячейка с SCARA, RGB-perception, кинематикой, планировщиком, FSM и контактной моделью. Кампания воспроизводима, но итог смешанной сортировки ниже заранее заданной цели: 21/30 вместо 27/30. Поэтому корректный вывод — отдельные полные виртуальные циклы достигнуты, а приемочная полнота не достигнута. Следующий технический шаг — анализ отказов захвата и учет заполнения слотов, затем повторная кампания; этот доклад не утверждает результаты физической сборки.\nИсточники: 08_Записка/source/Пояснительная_записка.md, заключение; 07_Испытания/Анализ.md; 07_Испытания/tables/Проверка_повторяемости_V14.json.");
}

if (presentation.slides.items.length !== 15) throw new Error(`Expected 15 slides, got ${presentation.slides.items.length}`);

const candidatePath = path.join(tempDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const finalizerDir = path.join(projectDir, ".m11_finalizer");
await fs.mkdir(finalizerDir, { recursive: true });
const finalizerReceipt = path.join(finalizerDir, "Презентация_к_защите.final.validation.json");
const result = await finalizePresentation({
  explicitTotalSlideCount: 15,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir: projectDir,
  candidatePath,
  finalPath: finalPptx,
  pythonExecutable: process.env.RUNTIME_PYTHON,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
  fontPolicy: { basis: "design", families: [fontFamily], scriptFonts: { cs: fontFamily } },
  verifyArtifactToolImport: true,
  receiptPath: finalizerReceipt,
});
console.log(JSON.stringify({ finalPptx, finalizerReceipt, result }, null, 2));

async function exists(filePath) {
  try { await fs.access(filePath); return true; } catch { return false; }
}

async function sha256(filePath) {
  const { createHash } = await import("node:crypto");
  const hash = createHash("sha256");
  hash.update(await fs.readFile(filePath));
  return hash.digest("hex");
}
