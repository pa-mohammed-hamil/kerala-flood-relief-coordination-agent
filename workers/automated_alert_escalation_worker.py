"""
Automated Alert Escalation Worker
Background worker that monitors conditions every 15 minutes and triggers automated escalation
"""
import time
import logging
from datetime import datetime
from backend.database import SessionLocal
from backend.services.automated_alert_escalation_service import ContinuousMonitoringService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Monitoring interval in seconds (15 minutes = 900 seconds)
MONITORING_INTERVAL = 900


def run_monitoring_cycle():
    """Run a single monitoring cycle for all districts"""
    db = SessionLocal()
    try:
        logger.info("=" * 80)
        logger.info("STARTING AUTOMATED ALERT MONITORING CYCLE")
        logger.info("=" * 80)
        
        monitoring_service = ContinuousMonitoringService(db)
        results = monitoring_service.monitor_all_districts()
        
        # Log summary
        logger.info(f"Districts monitored: {results['districts_monitored']}")
        logger.info(f"Alert levels: {results['alerts_by_level']}")
        
        if results['escalations']:
            logger.warning(f"ESCALATIONS: {len(results['escalations'])} alerts escalated")
            for escalation in results['escalations']:
                logger.warning(
                    f"  - {escalation['district']}: {escalation['level'].upper()} "
                    f"(Alert ID: {escalation['alert_id']})"
                )
        else:
            logger.info("No escalations in this cycle")
        
        logger.info("=" * 80)
        logger.info("MONITORING CYCLE COMPLETE")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in monitoring cycle: {e}", exc_info=True)
        raise
    finally:
        db.close()


def main():
    """Main worker loop"""
    logger.info("🚀 AUTOMATED ALERT ESCALATION WORKER STARTED")
    logger.info(f"Monitoring interval: {MONITORING_INTERVAL} seconds ({MONITORING_INTERVAL/60} minutes)")
    logger.info("Monitoring all 14 Kerala districts for:")
    logger.info("  • Real-time rainfall (IMD)")
    logger.info("  • River water levels (CWC)")
    logger.info("  • Dam storage levels (KSDMA)")
    logger.info("")
    logger.info("Alert Levels:")
    logger.info("  🟢 GREEN: Normal conditions")
    logger.info("  🟡 YELLOW: Watch - Be alert (Rainfall >50mm OR River >70% OR Dam >80%)")
    logger.info("  🟠 ORANGE: Warning - Prepare to evacuate (Rainfall >100mm OR River >85% OR Dam >90%)")
    logger.info("  🔴 RED: Critical - Evacuate immediately (Rainfall >150mm OR River >100% OR Dam >95%)")
    logger.info("")
    logger.info("Automated Actions:")
    logger.info("  • Community SMS/WhatsApp alerts")
    logger.info("  • Emergency coordinator notifications")
    logger.info("  • Automated voice calls (RED alerts)")
    logger.info("  • Rescue team activation")
    logger.info("  • Shelter opening")
    logger.info("  • Resource mobilization")
    logger.info("")
    
    cycle_count = 0
    
    while True:
        try:
            cycle_count += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"CYCLE #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}\n")
            
            # Run monitoring
            results = run_monitoring_cycle()
            
            # Log next run time
            next_run = datetime.now().replace(
                minute=(datetime.now().minute // 15 + 1) * 15 % 60,
                second=0,
                microsecond=0
            )
            logger.info(f"\n⏰ Next monitoring cycle: {next_run.strftime('%H:%M:%S')}")
            
            # Sleep until next interval
            time.sleep(MONITORING_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("\n🛑 Worker stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            logger.info(f"Retrying in 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    main()
