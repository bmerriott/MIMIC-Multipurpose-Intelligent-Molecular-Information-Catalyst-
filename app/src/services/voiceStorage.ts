/**
 * Local Voice Storage Service
 * Stores voice creation audio files in IndexedDB (not localStorage)
 * IndexedDB has 1GB+ capacity vs 5-10MB for localStorage
 */

const DB_NAME = 'MimicAI_Voices';
const DB_VERSION = 2; // Bump version for new store name
const STORE_NAME = 'voice_creates';

interface VoiceCreateRecord {
  id: string; // persona_id
  audio_data: string; // base64
  reference_text?: string;
  created_at: string;
}

class VoiceStorage {
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log('[VoiceStorage] Opening IndexedDB:', DB_NAME, 'version:', DB_VERSION);
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        console.error('[VoiceStorage] Failed to open IndexedDB:', request.error);
        reject(request.error);
      };
      request.onsuccess = () => {
        this.db = request.result;
        console.log('[VoiceStorage] IndexedDB opened successfully, stores:', Array.from(this.db.objectStoreNames));
        resolve();
      };

      request.onupgradeneeded = (event) => {
        console.log('[VoiceStorage] Database upgrade needed, old version:', event.oldVersion, 'new version:', DB_VERSION);
        const db = (event.target as IDBOpenDBRequest).result;
        const oldStoreName = 'voice_creates';
        
        // Create new store
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          console.log('[VoiceStorage] Creating object store:', STORE_NAME);
          db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        }
        
        // Migrate data from old store if it exists
        if (db.objectStoreNames.contains(oldStoreName)) {
          console.log('[VoiceStorage] Found old store, migrating data from:', oldStoreName);
          const oldTransaction = (event.target as IDBOpenDBRequest).transaction;
          if (oldTransaction) {
            const oldStore = oldTransaction.objectStore(oldStoreName);
            const cursorRequest = oldStore.openCursor();
            
            cursorRequest.onsuccess = (cursorEvent) => {
              const cursor = (cursorEvent.target as IDBRequest).result;
              if (cursor) {
                const record = cursor.value;
                console.log('[VoiceStorage] Migrating record:', record.id);
                
                // Save to new store
                const newStore = oldTransaction.objectStore(STORE_NAME);
                const newRecord: VoiceCreateRecord = {
                  id: record.id,
                  audio_data: record.audio_data,
                  reference_text: record.reference_text,
                  created_at: record.created_at || new Date().toISOString(),
                };
                newStore.put(newRecord);
                
                cursor.continue();
              } else {
                console.log('[VoiceStorage] Migration complete');
              }
            };
          }
        }
      };
    });
  }

  async saveVoice(personaId: string, audioData: string, referenceText?: string): Promise<void> {
    console.log('[VoiceStorage] Saving voice for personaId:', personaId, 'audio length:', audioData?.length);
    if (!this.db) await this.init();
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      
      const record: VoiceCreateRecord = {
        id: personaId,
        audio_data: audioData,
        reference_text: referenceText,
        created_at: new Date().toISOString(),
      };

      const request = store.put(record);
      request.onerror = () => {
        console.error('[VoiceStorage] Save failed:', request.error);
        reject(request.error);
      };
      request.onsuccess = () => {
        console.log('[VoiceStorage] Save successful for personaId:', personaId);
        resolve();
      };
    });
  }

  async getVoice(personaId: string): Promise<VoiceCreateRecord | null> {
    console.log('[VoiceStorage] Loading voice for personaId:', personaId);
    try {
      if (!this.db) await this.init();
      
      return new Promise((resolve, reject) => {
        try {
          const transaction = this.db!.transaction([STORE_NAME], 'readonly');
          const store = transaction.objectStore(STORE_NAME);
          const request = store.get(personaId);

          request.onerror = () => {
            console.error('[VoiceStorage] Load failed:', request.error);
            reject(request.error);
          };
          request.onsuccess = () => {
            const result = request.result;
            if (result) {
              console.log('[VoiceStorage] Load result: found (audio:', result.audio_data?.length, 'chars)');
              resolve(result);
            } else {
              // Fallback: check old store name for migration
              const oldStoreName = 'voice_creates';
              if (this.db!.objectStoreNames.contains(oldStoreName)) {
                console.log('[VoiceStorage] Not found in new store, checking old store:', oldStoreName);
                try {
                  const oldTransaction = this.db!.transaction([oldStoreName], 'readonly');
                  const oldStore = oldTransaction.objectStore(oldStoreName);
                  const oldRequest = oldStore.get(personaId);
                  
                  oldRequest.onsuccess = () => {
                    const oldResult = oldRequest.result;
                    if (oldResult) {
                      console.log('[VoiceStorage] Found in old store, audio length:', oldResult.audio_data?.length);
                      // Also migrate it to new store for next time
                      this.migrateRecord(oldResult).catch(e => console.error('[VoiceStorage] Migration failed:', e));
                    } else {
                      console.log('[VoiceStorage] Not found in old store either');
                    }
                    resolve(oldResult || null);
                  };
                  
                  oldRequest.onerror = () => {
                    console.error('[VoiceStorage] Old store lookup failed:', oldRequest.error);
                    resolve(null);
                  };
                } catch (oldErr) {
                  console.error('[VoiceStorage] Old store transaction error:', oldErr);
                  resolve(null);
                }
              } else {
                console.log('[VoiceStorage] Load result: not found (no old store)');
                resolve(null);
              }
            }
          };
        } catch (err) {
          console.error('[VoiceStorage] Transaction error:', err);
          reject(err);
        }
      });
    } catch (err) {
      console.error('[VoiceStorage] Init error:', err);
      return null;
    }
  }

  async deleteVoice(personaId: string): Promise<void> {
    if (!this.db) await this.init();
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(personaId);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }

  private async migrateRecord(oldRecord: any): Promise<void> {
    console.log('[VoiceStorage] Migrating record to new store:', oldRecord.id);
    const newRecord: VoiceCreateRecord = {
      id: oldRecord.id,
      audio_data: oldRecord.audio_data,
      reference_text: oldRecord.reference_text,
      created_at: oldRecord.created_at || new Date().toISOString(),
    };
    
    const transaction = this.db!.transaction([STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    await new Promise<void>((resolve, reject) => {
      const request = store.put(newRecord);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        console.log('[VoiceStorage] Migration successful for:', oldRecord.id);
        resolve();
      };
    });
  }

  async getStorageSize(): Promise<number> {
    if (!this.db) await this.init();
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const records: VoiceCreateRecord[] = request.result;
        let totalSize = 0;
        records.forEach(record => {
          // Approximate size: base64 is ~4/3 of binary
          totalSize += record.audio_data.length * 0.75;
        });
        resolve(totalSize);
      };
    });
  }

  async getStorageInfo(): Promise<{ used: number; quota: number; usage: number }> {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      const used = estimate.usage || 0;
      const quota = estimate.quota || 0;
      return {
        used,
        quota,
        usage: quota > 0 ? (used / quota) * 100 : 0,
      };
    }
    return { used: 0, quota: 0, usage: 0 };
  }
}

export const voiceStorage = new VoiceStorage();
