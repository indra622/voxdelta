import type { PrivacyPosture } from "./posture";
import { describeTransfer } from "./transfer";

export function privacyBadgeText(posture: PrivacyPosture): string {
  if (posture.kind === "local") return "Local-only · verified";
  if (posture.kind === "remote") return "Remote stage detected";
  if (posture.kind === "unknown") return "Local-only · unverified";
  return "Checking providers";
}

export function PrivacyNote({ posture }: { posture: PrivacyPosture }) {
  if (posture.kind === "checking") {
    return <p className="privacy-note">로컬 전용 여부를 백엔드 설정에서 확인하는 중입니다.</p>;
  }
  if (posture.kind === "unknown") {
    return (
      <p className="privacy-note warning" role="status">
        백엔드 provider 설정을 읽지 못했습니다. 지금은 로컬 전용이라고 보증할 수 없습니다.
      </p>
    );
  }
  if (posture.kind === "remote") {
    return (
      <p className="privacy-note warning" role="status">
        로컬 전용이 아닙니다. 다음 단계가 데이터를 외부로 보내도록 설정돼 있습니다 —{" "}
        {posture.transfers.map(describeTransfer).join(" · ")}.
      </p>
    );
  }
  return (
    <p className="privacy-note">
      파이프라인 {posture.stageCount}단계 모두 로컬 모델로 설정돼 있음을 백엔드에서 확인했습니다. 오디오, 전사문, 결과가 외부로 전송되지 않습니다.
      다만 분석하는 동안 이 컴퓨터의 로컬 폴더에는 저장되며, 새 분석을 시작하거나 삭제하면 지워집니다.
    </p>
  );
}
