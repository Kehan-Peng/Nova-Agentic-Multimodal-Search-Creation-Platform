import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";

const videoPath = process.env.CAMCAT_E2E_VIDEO;
const imagePath = process.env.CAMCAT_E2E_IMAGE;
const apiBase = process.env.CAMCAT_E2E_API_BASE ?? "http://localhost:8000";
const userId = process.env.VITE_CAMCAT_USER_ID ?? "camcat-local-user";

test.skip(!videoPath || !imagePath || !existsSync(videoPath) || !existsSync(imagePath), "real E2E video and image fixtures are required");

test("real multimodal import, edit, rollback, conflict and render journey", async ({ page, request }) => {
  let latestSession: { editing_session_id: string; state_version: number } | undefined;
  page.on("response", async (response) => {
    if (response.url().endsWith("/api/v1/editing/sessions") && response.request().method() === "POST" && response.ok()) {
      latestSession = await response.json();
    }
  });

  await page.goto("/");
  await expect(page.getByText("CamCat", { exact: true }).first()).toBeVisible();

  // Upload controls live in the editor workspace, not on the project list.
  const openProject = page.locator('button[aria-label^="打开项目 "]').first();
  await expect(openProject).toBeVisible();
  await openProject.click();
  await expect(page.getByPlaceholder("有问题，尽管问")).toBeVisible();

  await page.locator('input[type="file"][accept="video/*"]').setInputFiles(videoPath!);
  await expect(page.getByText(/未写入素材库或向量库/)).toBeVisible({ timeout: 20 * 60 * 1000 });
  const enterEditing = page.getByRole("button", { name: "进入编辑计划", exact: true });
  if (await enterEditing.count()) {
    await enterEditing.click();
  }
  await expect(page.getByPlaceholder("有问题，尽管问")).toBeVisible();

  await page.locator('input[type="file"][accept="image/*"]').setInputFiles(imagePath!);
  await expect(page.getByText(imagePath!.split("/").pop()!)).toBeVisible();
  const query = page.getByPlaceholder("有问题，尽管问");
  await query.fill("寻找与参考图相似的有活力镜头，剪成 12 秒竖屏短片");
  await query.press("Enter");
  await expect(page.getByText(/剪辑计划已通过 State Patch 写入/)).toBeVisible({ timeout: 10 * 60 * 1000 });
  await expect(page.getByText("validate_patch", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: /Export/ }).click();
  await expect(page.getByText("视频已成功导出，可下载！")).toBeVisible({ timeout: 20 * 60 * 1000 });
  const download = page.locator('a[download]');
  await expect(download).toHaveAttribute("href", /^https?:\/\//);
  await expect(page.getByTestId("camcat-video-preview")).toHaveAttribute("src", /^https?:\/\//);
  expect(latestSession).toBeTruthy();

  await query.fill("节奏更紧凑，字幕更短");
  await query.press("Enter");
  await expect(page.getByText("v3", { exact: true })).toBeVisible({ timeout: 10 * 60 * 1000 });
  await page.getByRole("button", { name: /Rollback/ }).click();
  await expect(page.getByText("v4", { exact: true })).toBeVisible();

  const sessionId = latestSession!.editing_session_id;
  const externalPatch = await request.patch(`${apiBase}/api/v1/editing/sessions/${sessionId}`, {
    headers: { "X-User-Id": userId },
    data: {
      base_version: 4,
      operations: [{ op: "replace", path: "/goal", value: "另一个窗口的更新" }],
      reason: "E2E deliberate concurrent edit",
    },
  });
  expect(externalPatch.ok()).toBeTruthy();

  await query.fill("这是使用过期版本发起的编辑");
  await query.press("Enter");
  await expect(page.getByText(/编辑状态已被其他操作更新/)).toBeVisible({ timeout: 10 * 60 * 1000 });
});
