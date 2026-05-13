import { createReadingResponse } from '@/lib/agent-protocol';
import { rootLocale } from '@/lib/route-helpers';

export function GET() {
  return createReadingResponse(rootLocale);
}
