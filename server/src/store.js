import fs from 'node:fs/promises';
import path from 'node:path';

// Minimal JSON-file store for diagnosis history. Writes are serialised so
// concurrent requests can't clobber each other.
export class HistoryStore {
  constructor(file) {
    this.file = file;
    this.records = null;
    this.queue = Promise.resolve();
  }

  async load() {
    if (this.records) return this.records;
    try {
      this.records = JSON.parse(await fs.readFile(this.file, 'utf8'));
    } catch (err) {
      if (err.code !== 'ENOENT') throw err;
      this.records = [];
    }
    return this.records;
  }

  persist() {
    this.queue = this.queue.then(async () => {
      await fs.mkdir(path.dirname(this.file), { recursive: true });
      const tmp = `${this.file}.tmp`;
      await fs.writeFile(tmp, JSON.stringify(this.records, null, 2));
      await fs.rename(tmp, this.file);
    });
    return this.queue;
  }

  async list() {
    return [...(await this.load())].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }

  async get(id) {
    return (await this.load()).find((r) => r.id === id) ?? null;
  }

  async add(record) {
    (await this.load()).push(record);
    await this.persist();
    return record;
  }

  async remove(id) {
    const records = await this.load();
    const idx = records.findIndex((r) => r.id === id);
    if (idx === -1) return null;
    const [removed] = records.splice(idx, 1);
    await this.persist();
    return removed;
  }
}
