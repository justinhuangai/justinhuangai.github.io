import { createAgentHomeResponse } from '@/lib/agent-protocol';
import { rootLocale } from '@/lib/route-helpers';

export async function GET() {
  return createAgentHomeResponse(rootLocale);
}
