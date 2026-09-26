import type { Plugin } from "vite";
import fs from "node:fs/promises";
import path from "node:path";
import { slugifyStr } from "./slugify";

const THREADS_DIR = path.resolve(process.cwd(), "src/content/threads");

function parseFrontmatter(content: string) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/);
  if (!match) return { data: {}, body: content };

  const yamlStr = match[1];
  const body = match[2];
  const data: Record<string, any> = {};

  yamlStr.split("\n").forEach(line => {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) return;
    const key = line.slice(0, colonIdx).trim();
    let val = line.slice(colonIdx + 1).trim();

    if (!key) return;

    if (val === "null" || val === "") {
      data[key] = null;
    } else if (val === "true") {
      data[key] = true;
    } else if (val === "false") {
      data[key] = false;
    } else if (val.startsWith("[") && val.endsWith("]")) {
      try {
        data[key] = JSON.parse(val);
      } catch {
        data[key] = val
          .slice(1, -1)
          .split(",")
          .map(s => s.trim().replace(/^["']|["']$/g, ""));
      }
    } else if (val.startsWith('"') && val.endsWith('"')) {
      data[key] = val.slice(1, -1);
    } else if (val.startsWith("'") && val.endsWith("'")) {
      data[key] = val.slice(1, -1);
    } else {
      data[key] = val;
    }
  });

  return { data, body };
}

function stringifyFrontmatter(data: Record<string, any>): string {
  const lines = ["---"];
  for (const [key, value] of Object.entries(data)) {
    if (value === null || value === undefined) {
      lines.push(`${key}: null`);
    } else if (key === "pubDatetime" || key === "modDatetime") {
      lines.push(`${key}: ${value}`);
    } else if (typeof value === "string") {
      lines.push(`${key}: ${JSON.stringify(value)}`);
    } else if (typeof value === "boolean" || typeof value === "number") {
      lines.push(`${key}: ${value}`);
    } else if (Array.isArray(value)) {
      lines.push(`${key}: [${value.map(v => JSON.stringify(v)).join(", ")}]`);
    } else {
      lines.push(`${key}: ${JSON.stringify(value)}`);
    }
  }
  lines.push("---");
  return lines.join("\n");
}

function getJsonBody(req: any): Promise<any> {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk: any) => (body += chunk));
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

export function devThreadsPlugin(): Plugin {
  return {
    name: "vite-dev-threads-plugin",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url || "";
        // Match /api/threads or /LevitatingLabs/api/threads (ignoring query strings)
        const cleanUrl = url.split("?")[0];
        if (!cleanUrl.endsWith("/api/threads")) {
          return next();
        }

        res.setHeader("Content-Type", "application/json");

        try {
          if (req.method === "GET") {
            await fs.mkdir(THREADS_DIR, { recursive: true });
            const files = await fs.readdir(THREADS_DIR);
            const mdFiles = files.filter(f => f.endsWith(".md") || f.endsWith(".mdx"));

            const threads = await Promise.all(
              mdFiles.map(async filename => {
                const filePath = path.join(THREADS_DIR, filename);
                const raw = await fs.readFile(filePath, "utf-8");
                const { data, body } = parseFrontmatter(raw);
                const id = filename.replace(/\.(md|mdx)$/, "");
                return { id, filename, data, body };
              })
            );

            res.statusCode = 200;
            return res.end(JSON.stringify(threads));
          }

          if (req.method === "POST") {
            const body = await getJsonBody(req);
            const {
              title,
              status = "in-progress",
              description = "",
              tags = ["experiment"],
              relatedArticle = null,
              featured = false,
              initialNoteTitle = "Day 1: Setup & Goals",
              initialNoteBody = "",
            } = body;

            if (!title) {
              res.statusCode = 400;
              return res.end(JSON.stringify({ error: "Title is required" }));
            }

            const slug = slugifyStr(title);
            const filename = `${slug}.md`;
            const filePath = path.join(THREADS_DIR, filename);

            const nowIso = new Date().toISOString();
            const dateFormatted = new Date().toLocaleDateString("en-GB", {
              day: "numeric",
              month: "short",
              year: "numeric",
            });

            const frontmatterObj = {
              title,
              pubDatetime: nowIso,
              modDatetime: nowIso,
              status,
              description,
              tags: Array.isArray(tags) ? tags : String(tags).split(",").map((t: string) => t.trim()),
              relatedArticle: relatedArticle || null,
              featured: Boolean(featured),
            };

            const frontmatterText = stringifyFrontmatter(frontmatterObj);

            const contentText = `${frontmatterText}

## ${initialNoteTitle || "Day 1: Setup & Goals"}
*${dateFormatted}*

${initialNoteBody || "Initial experiment setup and configuration."}
`;

            await fs.mkdir(THREADS_DIR, { recursive: true });
            await fs.writeFile(filePath, contentText, "utf-8");

            res.statusCode = 201;
            return res.end(JSON.stringify({ success: true, id: slug, filename }));
          }

          if (req.method === "PUT") {
            const body = await getJsonBody(req);
            const { id, action = "append_note", status, relatedArticle, noteTitle, noteBody } = body;

            if (!id) {
              res.statusCode = 400;
              return res.end(JSON.stringify({ error: "Thread ID is required" }));
            }

            const filename = id.endsWith(".md") || id.endsWith(".mdx") ? id : `${id}.md`;
            const filePath = path.join(THREADS_DIR, filename);

            const raw = await fs.readFile(filePath, "utf-8");
            const { data, body: fileBody } = parseFrontmatter(raw);

            const nowIso = new Date().toISOString();
            const dateFormatted = new Date().toLocaleDateString("en-GB", {
              day: "numeric",
              month: "short",
              year: "numeric",
            });

            data.modDatetime = nowIso;
            if (status) data.status = status;
            if (relatedArticle !== undefined) data.relatedArticle = relatedArticle || null;

            let updatedBody = fileBody;

            if (action === "append_note" && (noteBody || noteTitle)) {
              const titleText = noteTitle || "Timeline Update";
              const noteSection = `\n\n---\n\n## ${titleText}\n*${dateFormatted}*\n\n${noteBody || ""}`;
              updatedBody = fileBody.trimEnd() + noteSection;
            }

            const frontmatterText = stringifyFrontmatter(data);
            const fullText = `${frontmatterText}\n${updatedBody.startsWith("\n") ? updatedBody : `\n${updatedBody}`}`;

            await fs.writeFile(filePath, fullText, "utf-8");

            res.statusCode = 200;
            return res.end(JSON.stringify({ success: true, id }));
          }

          if (req.method === "DELETE") {
            const body = await getJsonBody(req);
            const { id } = body;
            if (!id) {
              res.statusCode = 400;
              return res.end(JSON.stringify({ error: "Thread ID is required" }));
            }

            const filename = id.endsWith(".md") || id.endsWith(".mdx") ? id : `${id}.md`;
            const filePath = path.join(THREADS_DIR, filename);

            await fs.unlink(filePath);

            res.statusCode = 200;
            return res.end(JSON.stringify({ success: true, id }));
          }

          res.statusCode = 405;
          return res.end(JSON.stringify({ error: "Method not allowed" }));
        } catch (err: any) {
          res.statusCode = 500;
          return res.end(JSON.stringify({ error: err.message || "Internal server error" }));
        }
      });
    },
  };
}
