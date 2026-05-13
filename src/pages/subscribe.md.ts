import { createSubscribeResponse } from '@/lib/agent-protocol';
import { rootLocale } from '@/lib/route-helpers';

export function GET() {
  return createSubscribeResponse(rootLocale);
}
